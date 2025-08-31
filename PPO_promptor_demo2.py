import matplotlib
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import gym  # Import gym for creating and managing environments
import numpy as np  # Import numpy for numerical computations

class TokenEnv(gym.Env):
    def __init__(self, token_list, token_scores, max_steps=None):
        super().__init__()
        self.token_list = token_list  # List of tokens
        self.token_scores = token_scores  # Mapping of tokens to scores
        self.n = len(token_list)  # Number of tokens
        self.observation_space = gym.spaces.MultiBinary(self.n)  # One-hot encoded observation space, size = number of tokens
        self.action_space = gym.spaces.Discrete(self.n)  # Discrete action space, size = number of tokens
        self.current_step = 0  # Current step
        self.used_tokens = set()  # Set of used tokens
        self.max_steps = max_steps or self.n  # Max steps (default = number of tokens)
        self.reset()  # Reset the environment
    
    def reset(self, *, seed=None, options=None):
        self.used_tokens = set()  # Reset used tokens
        self.current_step = 0  # Reset step counter
        observation = np.zeros(self.n, dtype=np.int32)  # Create an all-zero observation array
        return observation, {}  # Return observation and extra info
        
    def step(self, action):
        token = self.token_list[action]  # Choose token by action
        reward = 0  # Initialize reward
        terminated = False  # Initialize termination flag
        truncated = False  # Initialize truncation flag
        if token in self.used_tokens:  # If token already used
            reward = -1  # Penalty for duplicate token
        else:
            reward = self.token_scores[token]  # Reward = token score
            self.used_tokens.add(token)  # Add token to used set
            self.current_step += 1  # Increment step counter
        observation = np.zeros(self.n, dtype=np.int32)  # Create an all-zero observation
        for idx, t in enumerate(self.token_list):
            if t in self.used_tokens:
                observation[idx] = 1  # Mark used tokens as 1
        if self.current_step >= self.max_steps or len(self.used_tokens) == self.n:  
            terminated = True  # Terminate if max steps reached or all tokens used
        return observation, reward, terminated, truncated, {}  # Return observation, reward, termination, truncation, extra info

# Example token list and scores
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
from tqdm import *  # For progress bar display

class PolicyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyModel, self).__init__()
        
        # Fully connected neural network with ReLU activations
        # Ends with a Softmax layer to produce probability distribution
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim =-1)
        )

    # Forward pass to output action probabilities
    def forward(self, x):
        action_prob = self.fc(x)
        return action_prob

# Value model: estimates the value given a state
class ValueModel(nn.Module):
    def __init__(self, input_dim):
        super(ValueModel, self).__init__()
        
        # Similar to policy model but outputs a single value
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    # Forward pass to estimate value
    def forward(self, x):
        value = self.fc(x)
        return value

# PPO class
class PPO:
    # Constructor with environment and hyperparameters
    def __init__(self, env, learning_rate=0.001, gamma=0.99, lamda=0.95, clip_eps=0.2, epochs=10):
        self.env = env
        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.epochs = epochs

        # Detect available device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy and value models
        self.policy_model = PolicyModel(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.value_model = ValueModel(env.observation_space.shape[0]).to(self.device)

        # Adam optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=learning_rate)
    
    # Use policy model to sample actions
    def choose_action(self, state):
        # Convert state to tensor
        state = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            action_prob = self.policy_model(state)
        
        # Sample from action distribution
        c = torch.distributions.Categorical(action_prob)
        action = c.sample()
        return action.item()
    
    # Generalized Advantage Estimation (GAE)
    def calc_advantage(self, td_delta):
        td_delta = td_delta.cpu().detach().numpy()  # Convert to numpy
        advantage = 0
        advantage_list = []
        # Backward pass for advantage calculation
        for r in td_delta[::-1]:
            advantage = r + self.gamma * self.lamda * advantage
            advantage_list.insert(0, advantage)
        return torch.FloatTensor(np.array(advantage_list)).to(self.device)
    
    # Update policy and value networks
    def update(self, buffer):
        # Extract transition tuples
        states, actions, rewards, next_states, dones = zip(*buffer)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(self.device)

        with torch.no_grad():
            # Old action probabilities
            old_action_prob = torch.log(self.policy_model(states).gather(1, actions))
            
            # TD target and TD error
            td_target = rewards + (1 - dones) * self.gamma * self.value_model(next_states)
            td_delta = td_target - self.value_model(states)
        
        # Advantage estimation
        advantage = self.calc_advantage(td_delta)
        
        # Multiple epochs update
        for i in range(self.epochs):
            # New action probabilities
            action_prob = torch.log(self.policy_model(states).gather(1, actions))
            # Probability ratio
            ratio = torch.exp(action_prob - old_action_prob)
            
            # Clipped surrogate objective
            part1 = ratio * advantage
            part2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            policy_loss = -torch.min(part1, part2).mean()
            value_loss = F.mse_loss(self.value_model(states), td_target).mean()
            
            # Backpropagation and parameter updates
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()


# Hyperparameters
max_episodes = 1000  # Number of training episodes
max_steps = 500  # Max steps per episode


from torch.utils.tensorboard import SummaryWriter
import gradio as gr

def run_ppo_demo(token_string, score_string, episodes):
    token_list = [t.strip() for t in token_string.split(",") if t.strip()]  # Process input token string
    score_list = [float(s) for s in score_string.split(",") if s.strip()]  # Process input score string
    token_scores = {t : s for t, s in zip(token_list, score_list)}  # Map tokens to scores
    env = TokenEnv(token_list, token_scores)  # Create environment
    agent = PPO(env)  # Create PPO agent
    episode_rewards = []  # Store rewards for each episode
    best_tokens = []  # Store best tokens selected
    best_rewards = float('-inf')  # Initialize best reward
    writer = SummaryWriter()  # TensorBoard logger
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
        writer.add_scalar("Episode Reward", episode_reward, episode)  # Log episode reward
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
