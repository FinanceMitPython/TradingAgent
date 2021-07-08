import gym
import gym_anytrading

import yfinance
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

ticker = yfinance.Ticker("AAPL")

df = ticker.history(period="1y")

class Agent:
    def __init__(self, env):
        self.env = env
        self.state = self.env.frame_bound[0]
        self.actions = self.env.action_space.n
        self.model = self.get_model()
        
    def get_model(self):
        model = Sequential()
        model.add(Dense(units=100, input_dim=self.state))
        model.add(Activation("relu"))
        model.add(Dense(units=self.actions)) # Output: Action [L, R]
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model
    
    def get_action(self, state):
        state = state.reshape(1, -1)
        action = self.model(state).numpy()[0]
        action = np.random.choice(self.actions, p=action)
        return action
    
    def get_samples(self, num_episodes: int):
        rewards = [0.0 for i in range(num_episodes)]
        episodes = [[] for i in range(num_episodes)]

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                new_state, reward, done, info = self.env.step(action)
                total_reward = info["total_profit"]
                episodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards, episodes
    
    def filter_episodes(self, rewards, episodes, percentile):
        reward_bound = np.percentile(rewards, percentile)
        x_train, y_train = [], []
        for reward, episode in zip(rewards, episodes):
            if reward >= reward_bound:
                observation = [step[0].reshape(env.frame_bound[0]) for step in episode]
                action = [step[1] for step in episode]
                x_train.extend(observation)
                y_train.extend(action)
        x_train = np.asarray(x_train)
        y_train = to_categorical(y_train, num_classes=self.actions) # L = 0 => [1, 0]
        return x_train, y_train, reward_bound
    
    def train(self, percentile, num_iterations, num_episodes):
        for iteration in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(rewards, episodes, percentile)
            #print(x_train[0].shape)
            self.model.fit(x=x_train, y=y_train, verbose=0)
            reward_mean = np.mean(rewards)
            print(f"Reward mean: {reward_mean}, reward bound: {reward_bound}")
            if reward_mean > 2.0:
                break
    
    def trade(self, num_episodes: int, render: bool = True):
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                action = self.get_action(state)
                n_state, reward, done, info = self.env.step(action)
                if render:
                    if info["total_reward"] > 10:
                        env.render()
                if done:
                    print(info)
                    break

    def test_trade(self, num_episodes, render = True):
        for episode in range(num_episodes):
            env = gym.make("stocks-v0", df=df, frame_bound=(200,250), window_size=5)
            state = env.reset()
            while True:
                action = self.get_action(state)
                n_state, reward, done, info = env.step(action)
                env.render()
                if done:
                    print(info)
                    break

if __name__ == "__main__":
    env = gym.make("stocks-v0", df=df, frame_bound=(10,200), window_size=5)
    agent = Agent(env)
    agent.train(percentile = 70.0, num_iterations=10, num_episodes=100)
    agent.trade(num_episodes = 10, render=False)
    agent.test_trade(num_episodes = 10)