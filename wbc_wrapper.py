import gymnasium as gym

import numpy as np
from qpsolvers import solve_qp

import mujoco

## A wrapper that lets the agent learn in task space (with the help of a WBC controller)
class WbcWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        ## Define this wrapper's action space as the desired task space;
        ## specifically, 6dog body accel
        action_task_space_dim = 6

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_task_space_dim,), dtype=np.float32)

    def step(self, action_task_space):

        ## Map task-space action to joint-space action using WBC
        action_joint_space = self.solve_qp(action_task_space)

        ## Step the environment as usual
        obs, reward, done, truncated, info = self.env.step(action_joint_space)
        return obs, reward, done, truncated, info

    def solve_qp(self, action_task_space):

        ## Number of decision variables
        n_dc = self.model.nv + self.model.nu + 3
        # print(f"nv: {self.model.nv}, nu: {self.model.nu}, n_dc: {n_dc}")
        assert n_dc == 15

        ## Assemble the equality constraint
        M = np.zeros((self.env.model.nv, self.env.model.nv))
        mujoco.mj_fullM(self.env.model, M, self.env.data.qM)

        bias_vector = self.env.data.qfrc_bias


        wheel_body_id = mujoco.mj_name2id(self.env.model, 1, "wheel")
        wheel_in_contact = False
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            if(contact.geom1 == wheel_body_id or contact.geom2 == wheel_body_id):
                wheel_in_contact = True
                contact_pos = contact.pos
                print("\n wheel in contact\n")
                break

        J_c = np.zeros((3, self.env.model.nv))
        assert self.env.model.nv == 9
        if wheel_in_contact:
            ## 
            J_c_pos = np.zeros((3, self.env.model.nv))
            J_c_rot = np.zeros((3, self.env.model.nv))

            human_body_id = mujoco.mj_name2id(self.env.model, 1, "human")
            mujoco.mj_jac(self.env.model, self.env.data, J_c_pos, J_c_rot, contact_pos, human_body_id)

            # J_c = np.block([
            #     J_c_pos.reshape(3, self.env.model.nv),
            #     J_c_rot.reshape(3, self.env.model.nv)
            # ])

            J_c = J_c_pos.reshape(3, self.env.model.nv)





        B = np.vstack([
            np.zeros((6, 3)),
            np.eye(3)
        ])

        if wheel_in_contact:

            # print(f"J_c: {J_c.shape}")

            A_eq = np.block([
                [M,   -B,               -J_c.T          ],
                [J_c, np.zeros((3, 3)), np.zeros((3, 3))]
            ])
            # print(f"bias_vector: {bias_vector.shape}")

            b_eq = np.block([
                [bias_vector.reshape(-1, 1)],
                [np.zeros((3, 1))]
            ])
            ## TODO: might have to make this a *x1 instead of *,
        else:
            # A = np.vstack([
            #     [M, -B, np.zeros_like(J_c.T)]
            # ])
            A_eq = np.block([[M, -B, np.zeros_like(J_c.T)]])
            # print(f"just made A_eq {A_eq.shape}")

            b_eq = np.block([
                [bias_vector.reshape(-1, 1)]
            ])


        ## Assemble the inequality constraint
        mu = 0.8
        min_normal_force = 0.5
        ## TODO: change tau max
        tau_max_hop = 200.0
        tau_max_twist = 200.0
        tau_max_roll = 200.0

        mu_over_root2 = (np.sqrt(2) / 2) * mu
        G_f = np.array([
            [-1,  0, -mu_over_root2],
            [ 1,  0, -mu_over_root2],
            [ 0, -1, -mu_over_root2],
            [ 0,  1, -mu_over_root2],
            [ 0,  0, -1            ]
        ])
        h_f = np.array([[0],
                        [0],
                        [0],
                        [0],
                        [-min_normal_force]])

        ## Input limit constraint
        G_u = np.array([
            [ 1,  0,  0],
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0, -1,  0],
            [ 0,  0,  1],
            [ 0,  0, -1]
        ])
        h_u = np.array([[tau_max_twist],
                        [tau_max_twist],
                        [tau_max_hop],
                        [tau_max_hop],
                        [tau_max_roll],
                        [tau_max_roll]])

        if wheel_in_contact:
            G = np.block([
                [np.zeros((6, 9)), G_u,              np.zeros((6, 3))],
                [np.zeros((5, 9)), np.zeros((5, 3)), G_f             ]
            ])



            h = np.block([
                [h_u],
                [h_f]
            ])

            # print(f"\nG: {G}")
            # print(f"\nh: {h}")

        else:
            G = np.block([
                [np.zeros((6, 9)), G_u,              np.zeros((6, 3))]
            ])
            h = np.block([
                [h_u]
            ])

            # print(f"\nG: {G}")
            # print(f"\nh: {h}")


        

        ## Get task space command
        cmd = np.zeros(n_dc)
        # print(f"action_task_space: {action_task_space}")
        cmd[:6] = action_task_space # TODO: for now action_task_space is simply 6dof body accel

        ## Task weights
        W = np.eye(n_dc) # TODO: these are inputs, or are they?

        ## Task jacobian
        J_t = np.eye(n_dc)

        Q = 2 * J_t.T @ W @ J_t
        Q += np.eye(n_dc) * 1e-6

        b = -2 * cmd.T @ W @ J_t

        # M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
        
        # P = M.T @ M  # this is a positive definite matrix
        # q = np.array([3.0, 2.0, 3.0]) @ M
        # G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        # h = np.array([3.0, 2.0, -2.0])
        # A = np.array([1.0, 1.0, 1.0])
        # b = np.array([1.0])

        print(f"A_eq {A_eq.shape}, b {b_eq.shape}, G {G.shape}, h {h.shape}")

        optimized_decision_variables = solve_qp(Q, b, G, h, A_eq, b_eq, solver="proxqp", verbose=True)
        # print("QP solution: ", optimized_decision_variables)
        if optimized_decision_variables is None:
            print("QP solver failed to find a solution.")
        # if optimized_decision_variables is None:
        #     print("QP Solver failed to find a solution.")
        #     print("A_eq:", A_eq)
        #     print("b_eq:", b_eq)
        #     print("G:", G)
        #     print("h:", h)
        #     return None

        # print(f"QP solution: {optimized_decision_variables = }")
        action_joint_space = optimized_decision_variables[self.model.nv:self.model.nv+self.model.nu]

        ## Convert from float64 to float32
        return action_joint_space.astype(np.float32)
