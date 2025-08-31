import matplotlib
import matplotlib.pyplot as plt # 导入matplotlib库，用于绘制图像
import gym # 导入gym库，用于创建和管理环境
import numpy as np # 导入numpy库，用于数值计算
class TokenEnv(gym.Env):
    def __init__(self, token_list, token_scores, max_steps=None):
        super().__init__()
        self.token_list = token_list # token列表
        self.token_scores =token_scores # token对应分数
        self.n = len(token_list) # token数量
        self.observation_space = gym.spaces.MultiBinary(self.n) # one-hot编码的观测空间，大小为token数量
        self.action_space = gym.spaces.Discrete(self.n) # 动作空间为离散空间，大小为token数量
        self.current_step = 0 # 当前步数
        self.used_tokens = set() # 已使用的token集合
        self.max_steps = max_steps or self.n # 最大步数默认为token数量
        self.reset() # 重置环境
    
    def reset(self, *, seed=None, options=None):
        self.used_tokens = set() # 重置已使用的token集合
        self.current_step = 0 # 重置当前步数
        observation = np.zeros(self.n, dtype=np.int32) # 创建一个全0的观测数组
        return observation, {} # 返回观测数组和额外信息
        
    def step(self, action):
        token = self.token_list[action] # 根据动作选择对应token
        reward = 0 # 初始化奖励为0
        terminated = False # 初始化终止状态为False
        truncated = False # 初始化截断状态为False
        if token in self.used_tokens: # 如果token已被使用
                reward = -1 # 奖励为-1
        else:
            reward = self.token_scores[token] # 奖励为token对应分数
            self.used_tokens.add(token) # 将token加入已使用集合
            self.current_step += 1 # 增加当前步数
        observation = np.zeros(self.n, dtype=np.int32) # 创建一个全0的观测数组
        for idx, t in enumerate(self.token_list):
                if t in self.used_tokens:
                    observation[idx] = 1 # 如果token已被使用，则在观测数组中对应位置为1
        if self.current_step >= self.max_steps or len(self.used_tokens) == self.n: # 如果达到最大步数或所有token都已使用
                terminated = True # 达到终止条件
        return observation, reward, terminated, truncated,{} # 返回观测数组、奖励、终止状态、截断状态和额外信息
        # 创建一个token列表和对应分数
token_list = ["1girl", "solo", "long hair", "hand fan", "twintails", "butterfly", "bugs", "very long hair", "smile", "kimono" ]
token_scores = {"1girl": 2, "solo": 1, "long hair":-1, "hand fan":-1, "twintails": 2, "butterfly": 0.5, "bugs": -2, "very long hair": 0.1, "smile": 1, "kimono": 3}
env = TokenEnv(token_list, token_scores)
        


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import sys
import time
import random
import collections
from tqdm import * # 用于显示进度条
class PolicyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyModel, self).__init__()
        
        # 使用全连接层构建一个简单的神经网络，ReLU作为激活函数
        # 最后加一个Softmax层，使得输出可以看作是概率分布
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim =-1)
        )

    # 定义前向传播，输出动作概率分布
    def forward(self, x):
        action_prob = self.fc(x)
        return action_prob

# 价值模型，给定状态估计价值
class ValueModel(nn.Module):
    def __init__(self, input_dim):
        super(ValueModel, self).__init__()
        
        # 网络结构和策略模型类似，但是输出层只有一个节点
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    # 定义前向传播，输出价值估计
    def forward(self, x):
        value = self.fc(x)
        return value

# 定义PPO类
class PPO:
    # 构造函数，参数包含环境，学习率，折扣因子，优势计算参数，clip参数，训练轮数
    def __init__(self, env, learning_rate=0.001, gamma=0.99, lamda=0.95, clip_eps=0.2, epochs=10):
        self.env = env
        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.epochs = epochs

        # 判断可用的设备是 CPU 还是 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 根据环境的观测空间和动作空间，定义策略模型和价值模型，并将模型移动到指定设备上
        self.policy_model = PolicyModel(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.value_model = ValueModel(env.observation_space.shape[0]).to(self.device)

        # 定义Adam优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=learning_rate)
    
    # 使用策略模型生成动作概率分布并采样
    def choose_action(self, state):
        # 将状态转换为tensor输入模型
        state = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            action_prob = self.policy_model(state)
        
        # 生成分布后采样返回动作
        c = torch.distributions.Categorical(action_prob)
        action = c.sample()
        return action.item()
    
    # 广义优势估计
    def calc_advantage(self, td_delta):
        # 将TD误差转换为numpy数组
        td_delta = td_delta.cpu().detach().numpy()
        # 初始化优势函数值及存储优势值的列表
        advantage = 0
        advantage_list = []
        # 反向遍历，从最后一步开始倒推
        for r in td_delta[::-1]:
            # 将当前步的TD误差及上一步优势加权值作为当前步的优势
            advantage = r + self.gamma * self.lamda * advantage
            # 将优势值加到列表开头，最终得到顺序序列
            advantage_list.insert(0, advantage)
        # 转换为tensor后返回
        return torch.FloatTensor(np.array(advantage_list)).to(self.device)
    
    # 模型更新
    def update(self, buffer):
        # 取出数据，并将其转换为numpy数组
        # 然后进一步转换为tensor，并将数据转移到指定计算资源设备上
        states, actions, rewards, next_states, dones = zip(*buffer)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(self.device)

        with torch.no_grad():
            # 计算旧策略下的动作概率
            old_action_prob = torch.log(self.policy_model(states).gather(1, actions))
            
            # 计算TD目标及误差
            td_target = rewards + (1 - dones) * self.gamma * self.value_model(next_states)
            td_delta = td_target - self.value_model(states)
        
        # 优势估计
        advantage = self.calc_advantage(td_delta)
        
        # 多步更新策略
        for i in range(self.epochs):
            # 计算新策略下的动作概率
            action_prob = torch.log(self.policy_model(states).gather(1, actions))
            # 计算策略动作概率比
            ratio = torch.exp(action_prob - old_action_prob)
            
            # CLIP修剪
            part1 = ratio * advantage
            part2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            # 计算策略损失
            policy_loss = -torch.min(part1, part2).mean()
            # 计算价值损失
            value_loss = F.mse_loss(self.value_model(states), td_target).mean()
            
            # 梯度清零、反向传播、更新参数
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()


# 定义超参数
max_episodes = 1000# 训练episode数
max_steps = 500 # 每个回合的最大步数

# 创建PPO对象
# agent = PPO(env)
# 定义保存每个回合奖励的列表
# episode_rewards = []

# 开始循环，tqdm用于显示进度条并评估任务时间开销
# for episode in tqdm(range(max_episodes), file=sys.stdout):
    # 重置环境并获取初始状态
   #  state, _ = env.reset()
    # 当前回合的奖励
   #  episode_reward = 0
    # 记录每个episode的信息
   #  buffer = []

    # 循环进行每一步操作
   #  for step in range(max_steps):
        # 根据当前状态选择动作
    #   action = agent.choose_action(state)
        # 执行动作，获取新的信息
    #     next_state, reward, terminated, truncated, info = env.step(action)
        # 判断是否达到终止状态
    #     done = terminated or truncated
        
        # 将这个五元组加到buffer中
    #     buffer.append((state, action, reward, next_state, done))
        # 累计奖励
    #     episode_reward += reward
        
        # 更新当前状态
    #     state = next_state

    #     if done:
    #         selected_tokens = [env.token_list[i] for i, used in enumerate(state) if used == 1]
    #         print(f"Episode {episode + 1}: Selected tokens: {selected_tokens}, Reward: {episode_reward}")
    #         break
    
    # 更新策略
   #  agent.update(buffer)
    # 记录当前回合奖励值
   #  episode_rewards.append(episode_reward)
    
    # 打印中间值
    # if episode % (max_episodes // 10) == 0:
      #   tqdm.write("Episode " + str(episode) + ": " + str(episode_reward))

from torch.utils.tensorboard import SummaryWriter
import gradio as gr
def run_ppo_demo(token_string,score_string,episodes):
    token_list = [t.strip() for t in token_string.split(",") if t.strip()] # 处理输入的token字符串
    score_list = [float(s) for s in score_string.split(",") if s.strip()] # 处理输入的分数字符串
    token_scores = {t : s for t, s in zip(token_list, score_list)} # 创建token和分数的映射
    env = TokenEnv(token_list, token_scores) # 创建环境
    agent = PPO(env) # 创建PPO代理
    episode_rewards = [] # 用于存储每个回合的奖励
    best_tokens = [] # 用于存储每个回合选择的最佳token
    best_rewards = float('-inf') # 初始化最佳奖励为负无穷大
    writer = SummaryWriter() # 创建TensorBoard记录器
    for episode in range(int(episodes)):
        state, _ = env.reset()
        episode_reward = 0
        buffer = []
        for step in range(len(token_list)):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.append((state, action, reward, next_state, done))
            episode_reward += reward
            state = next_state
            if done:
                selected_tokens = [env.token_list[i] for i, used in enumerate(state) if used == 1]
                if episode_reward > best_rewards:
                    best_tokens = selected_tokens
                    best_rewards = episode_reward
                break
        agent.update(buffer)
        writer.add_scalar("Episode Reward", episode_reward, episode) # 记录每个回合的奖励
    writer.close()
    return f"Best tokens: {', '.join(best_tokens)}\nBest reward: {best_rewards}"
 
app = gr.Interface(
    fn=run_ppo_demo,
    inputs=[
            gr.Textbox(label="Token list (comma separated)", value="1girl, solo, long hair, hand fan, twintails, butterfly, bugs, very long hair, smile, kimono"),
            gr.Textbox(label="Token scores (comma separated)", value="2, 1, -1, -1, 2, 0.5, -2, 0.1, 1, 3"),
            gr.Number(label="Episodes", value=1000, step=1, precision=0)
 ],
    outputs=gr.Textbox(label="Best tokens and rewards"),
    title="PPO Promptor Generator",
)

if __name__ == "__main__":
    app.launch()