import gym
from gym import spaces
import numpy as np

class DrivingEnv(gym.Env):
    def __init__(self):
        super(DrivingEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32)  # acceleration
        self.observation_space = spaces.Box(low=np.array([0.0, -10.0], dtype=np.float32), high=np.array([100.0, 10.0], dtype=np.float32), dtype=np.float32)  # [position, velocity]
        self.state = np.array([0.0, 0.0], dtype=np.float32)  # [position, velocity]
        self.goal_position = 100.0  # target position
        self.max_steps = 1000  # maximum steps per episode

    def reset(self):
        self.state = np.array([0.0, 0.0], dtype=np.float32)  # reset to initial position and velocity
        self.max_steps = 1000  # reset the maximum steps
        return self.state

    def step(self, action):
        acceleration = action[0]
        self.state[1] += acceleration  # update velocity
        self.state[0] += self.state[1]  # update position

        # Reward function
        distance_to_goal = abs(self.state[0] - self.goal_position)
        reward = -distance_to_goal  # negative reward for distance from goal

        # Done condition
        done = self.state[0] >= self.goal_position or abs(self.state[1]) > 10 or self.max_steps <= 0
        
        self.max_steps -= 1  # decrement step count

        return self.state, reward, done, {}

    def render(self):
        print(f"Current position: {self.state[0]:.2f}, Velocity: {self.state[1]:.2f}, Goal: {self.goal_position:.2f}")
