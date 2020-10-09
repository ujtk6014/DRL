import gym

env = gym.make('CartPole-v0')    # Cartpole定義
env.reset()    # Cartpoleの状態初期化

for i in range(100):
    env.render()    # Cartpoleのアニメーション
    observation, reward, done, info = env.step(env.action_space.sample())  # カートを動かし、その結果を返す
    print("Step:",i,done,"Reward:",reward,"Obs:",observation)