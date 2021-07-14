import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv

import yfinance
import numpy as np
import matplotlib.pyplot as plt
from finta import TA

from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

ticker = yfinance.Ticker("AMC")

df = ticker.history(period="1y")

df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)

df.dropna(inplace=True)

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low', 'SMA', 'RSI', 'OBV']].to_numpy()[start:end]
    return prices, signal_features

class CustomEnv(StocksEnv):
    _process_data = my_process_data

class Agent:
    def __init__(self, env):
        self.env = env
        self.state = 7
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
                total_reward += reward
                print(reward)
                print(total_reward)
                input()
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
                observation = [step[0].reshape(self.state) for step in episode]
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
            if reward_mean > 20:
                break
    
    def trade(self, num_episodes: int, render: bool = False):
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                action = self.get_action(state)
                n_state, reward, done, info = self.env.step(action)
                # if render:
                #     if info["total_reward"] > 10:
                #         env.render()
                if done:
                    print(info)
                    break

    # def test_trade(self, num_episodes, render = True):
    #     for episode in range(num_episodes):
    #         env = CustomEnv(df=df, window_size=1, frame_bound=(100,150))
    #         #env = gym.make("stocks-v0", df=df, frame_bound=(100,150), window_size=5)
    #         state = env.reset()
    #         while True:
    #             action = self.get_action(state)
    #             n_state, reward, done, info = env.step(action)
    #             env.render()
    #             if done:
    #                 print(info)
    #                 break

if __name__ == "__main__":
    env = CustomEnv(df=df, window_size=1, frame_bound=(10,100))
    #env = gym.make("stocks-v0", df=df, frame_bound=(10,100), window_size=5)
    agent = Agent(env)
    agent.train(percentile = 90.0, num_iterations=100, num_episodes=100)
    agent.trade(num_episodes = 10, render=False)
    # agent.test_trade(num_episodes = 10)
