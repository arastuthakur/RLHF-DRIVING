import numpy as np
import matplotlib.pyplot as plt
from environment.driving_env import DrivingEnv
from agent.dqn import DQNAgent

def train_agent(episodes=1000, batch_size=32):
    env = DrivingEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.shape[0])
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(np.array([action]))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")
        
        rewards.append(total_reward)
        agent.replay(batch_size)

    return rewards

if __name__ == "__main__":
    rewards = train_agent()
    
    # Plotting rewards
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards over Episodes')
    plt.show()
