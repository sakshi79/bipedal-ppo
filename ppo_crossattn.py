import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross-Attention Module
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, obs_seq, act_seq):
        # obs_seq: [B, L, D], act_seq: [B, L, D]
        Q = self.query(obs_seq)
        #Q = F.softmax(Q, dim=-1)
        K = self.key(act_seq)
        #K = F.softmax(K, dim=-1)
        V = self.value(act_seq)
        #V = F.softmax(V, dim=-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, L, L]
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)  # [B, L, D]
        return context + obs_seq

# Actor with Cross-Attention
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.fc_obs1 = nn.Linear(obs_dim, hidden_dim)
        self.fc_obs2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_act1 = nn.Linear(act_dim, hidden_dim)
        self.fc_act2 = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_seq, act_seq):
        #x = self.fc_obs1(obs_seq)
        x = torch.tanh(self.fc_obs1(obs_seq))
        #y = self.fc_act1(act_seq)
        y = torch.tanh(self.fc_act1(act_seq))
        z = self.cross_attn(x, y)  # [B, L, D]
        z = F.softmax(z, dim=-1)
        z = z[:, -1, :]  # use last context vector
        mu = self.fc_out(z)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

# Critic with Cross-Attention
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.fc_obs = nn.Linear(obs_dim, hidden_dim)
        self.fc_act = nn.Linear(act_dim, hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq, act_seq):
        x = torch.tanh(self.fc_obs(obs_seq))
        y = torch.tanh(self.fc_act(act_seq))
        z = self.cross_attn(x, y)
        z = z[:, -1, :]
        return self.fc_out(z)

# PPO Buffer

class Buffer:
    def __init__(self, obs_dim, act_dim, seq_len, buffer_size):
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Pre-allocate tensors on correct device
        self.obs = torch.zeros((buffer_size, seq_len, obs_dim), dtype=torch.float32, device=DEVICE)
        self.acts = torch.zeros((buffer_size, seq_len, act_dim), dtype=torch.float32, device=DEVICE)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32, device=DEVICE)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=DEVICE)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=DEVICE)
        self.logprobs = torch.zeros(buffer_size, dtype=torch.float32, device=DEVICE)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=DEVICE)
        
        self.idx = 0
        self.full = False

    def store(self, obs_seq, act_seq, action, reward, value, logprob, done):
        # Convert inputs to tensors if needed
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.tensor(obs_seq, dtype=torch.float32, device=DEVICE)
        if not isinstance(act_seq, torch.Tensor):
            act_seq = torch.tensor(act_seq, dtype=torch.float32, device=DEVICE)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=DEVICE)
        
        # Ensure value is scalar tensor
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.squeeze().item()
        
        # Store transition
        idx = self.idx % self.buffer_size
        self.obs[idx] = obs_seq
        self.acts[idx] = act_seq
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.logprobs[idx] = logprob.sum() if isinstance(logprob, torch.Tensor) else logprob
        self.dones[idx] = done
        
        self.idx += 1
        if self.idx >= self.buffer_size:
            self.full = True
            self.idx = 0

    def get(self):
        # Return only filled portion of buffer
        if self.full:
            return (
                self.obs,
                self.acts,
                self.actions,
                self.rewards,
                self.values,
                self.logprobs,
                self.dones
            )
        else:
            return (
                self.obs[:self.idx],
                self.acts[:self.idx],
                self.actions[:self.idx],
                self.rewards[:self.idx],
                self.values[:self.idx],
                self.logprobs[:self.idx],
                self.dones[:self.idx]
            )

    def clear(self):
        # Reset buffer
        self.idx = 0
        self.full = False
        # Clear tensors (keep allocated memory)
        self.obs.zero_()
        self.acts.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.logprobs.zero_()
        self.dones.zero_()

    def __len__(self):
        return self.buffer_size if self.full else self.idx

# PPO Trainer
class Trainer:
    def __init__(self, batch_size, actor_lr, critic_lr,
                 gamma=0.99, lam=0.95, clip=0.2, entropy_coef=0.01, epochs=10, seq_len=8, obs_dim=24, act_dim=4, hidden_dim=64):
        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(DEVICE)
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(DEVICE)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = Buffer(obs_dim, act_dim, seq_len, batch_size)
        self.seq_len = seq_len
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size

    def remember(self, *args):
        self.buffer.store(*args)

    def select_action(self, obs_seq, act_seq):
        with torch.no_grad():
            action_dist = self.actor(obs_seq, act_seq)
            value = self.critic(obs_seq, act_seq)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        return action, log_prob, value

    def compute_gae(self, rewards, values, dones):
        # Vectorized GAE computation
        with torch.no_grad():
            values_plus = torch.cat([values, torch.zeros(1, device=DEVICE)])
            deltas = rewards + self.gamma * values_plus[1:] * (1 - dones) - values_plus[:-1]
            
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
                advantages[t] = gae
                
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return advantages, returns

    def update(self):
        # Add these debug checks
        print(f"Shapes before update:")
        print(f"rewards: {rewards.shape}, values: {values.shape}, dones: {dones.shape}")
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Add gradient clipping
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        obs, acts, actions, rewards, values, logprobs, dones = self.buffer.get()
    
        # Ensure values is 1D
        values = values.view(-1)  # Flatten to [T]
        
        # Add bootstrap value with matching dimensions
        bootstrap = torch.zeros(1, device=values.device)
        values = torch.cat([values, bootstrap])  # Now both 1D
        
        # Compute GAE
        advs, rets = self.compute_gae(rewards, values, dones)
        
        for _ in range(self.epochs):
            for i in range(0, len(rewards), self.batch_size):
                sl = slice(i, i + self.batch_size)
                # No .to(DEVICE) needed if buffers already on device
                dist = self.actor(obs[sl].to(DEVICE), acts[sl].to(DEVICE))
                taken_actions = actions[sl].to(DEVICE)
                # Calculate logprobs for THOSE SPECIFIC ACTIONS
                new_logprob = dist.log_prob(taken_actions)  # Shape: [batch_size, action_dim=4]
                
                # Sum logprobs across action components
                new_logprob = new_logprob.sum(-1, keepdim=True)  # Shape: [batch_size, 1]
                
                # Get old logprobs (already stored as sums)
                old_logprob = logprobs[sl].to(DEVICE).unsqueeze(-1)  # Shape: [batch_size, 1]
                ratio = (new_logprob - old_logprob).exp()
                
                entropy = dist.entropy().mean()

                surr1 = ratio * advs[sl]
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advs[sl]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                value = self.critic(obs[sl], acts[sl]).squeeze()
                critic_loss = F.mse_loss(value, rets[sl])

                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

        self.buffer.clear()
