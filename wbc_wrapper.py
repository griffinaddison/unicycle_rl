import gym

## A wrapper that lets the agent learn in task space (with the help of a WBC controller)
class WbcWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        ## Define this warpper's action space as the desired task space;
        ## specifically, 12 = 6dof COM accel task(commands + weights)
        action_task_space_dim = 12

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_task_space_dim,), dtype=np.float32)

    def step(self, action_task_space):

        ## Map task-space action to joint-space action using WBC
        action_joint_space = self.solve_qp(action_task_space, self.env.state)

        ## Step the environment as usual
        obs, reward, done, info = self.env.step(action_joint_space)
        return obs, reward, done, info

    def solve_qp(self, high_level_action, state):

        ## TODO: Implement WBC heres

        return calculated_control_commands
