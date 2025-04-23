import torch
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from ppo_crossattn import Trainer

def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env

# Check if CUDA is available and set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)



def train_ppo(clip_ratio=0.2, gamma=0.99, gae_lambda=0.8, entropy_coef=1e-4, mini_batch_size=64,
              actor_lr=3e-5, critic_lr=4e-5, n_epochs=5, T=2048, n_episodes=1000, hardcore=False, seq_len=4):
    
    env = gym.make("BipedalWalker-v3", hardcore=hardcore, render_mode=None)
    seed = random.randint(0, 2**32 - 1)
    env = set_seed(seed, env)
    agent = Trainer(batch_size=mini_batch_size, actor_lr=actor_lr, critic_lr=critic_lr,
                 gamma=gamma, lam=gae_lambda, clip=clip_ratio, entropy_coef=entropy_coef, 
                 epochs=n_epochs, seq_len=seq_len)

    episode_rewards = []
    avg_rewards = []
    max_reward = -1000
    steps_at_maxr = 0

    for episode in (pbar := tqdm(range(n_episodes), desc="avg reward = N/A / max reward = N/A ")):
        steps = 0
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE).unsqueeze(0)
        action_seq = torch.zeros((1, seq_len, env.action_space.shape[0]), device=DEVICE)
        obs_seq = state.unsqueeze(1).repeat(1, seq_len, 1)

        terminated = truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            steps += 1

            action, prob, value = agent.select_action(obs_seq, action_seq)
            
            # Convert action to numpy for env.step()
            action_np = action.cpu().numpy().flatten()
            new_state, reward, terminated, truncated, _ = env.step(action_np)
            
            # Convert everything to proper tensors before storing
            new_state_tensor = torch.tensor(new_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            done_tensor = torch.tensor(terminated, dtype=torch.float32, device=DEVICE)
            
            # Store transition with properly converted tensors
            agent.remember(
                obs_seq.clone(),
                action_seq.clone(),
                action.clone(),
                reward_tensor,  # Now a tensor
                value.squeeze().clone(),  # Ensure scalar
                prob.clone(),
                done_tensor  # Now a tensor
            )

            episode_reward += reward

            # Update sequences
            obs_seq = torch.roll(obs_seq, shifts=-1, dims=1)
            obs_seq[:, -1] = new_state_tensor

            action_seq = torch.roll(action_seq, shifts=-1, dims=1)
            action_seq[:, -1] = action

            if len(agent.buffer.rewards) >= T:
                agent.update()
                agent.buffer.clear()

        episode_rewards.append(episode_reward)
        if episode_reward > max_reward:
            max_reward = episode_reward
            steps_at_maxr = steps
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        pbar.set_description(f"avg reward = {avg_reward:.2f} / max reward = {max_reward:.2f} / steps@max = {steps_at_maxr}")

    env.close()

    to_save = input("Save model? [y/N] ").lower() == 'y'
    if to_save:
        torch.save(agent.actor.state_dict(), "ppo_crossattn_actor.pt")
        torch.save(agent.critic.state_dict(), "ppo_crossattn_critic.pt")

    plt.plot(episode_rewards, label="Episode Reward")
    plt.plot(avg_rewards, label="Avg Reward (100)", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    if(to_save == 'Y' or to_save == 'y'):
        plt.savefig("ppo_crossattn_training_plot.png")
    plt.show()





train_ppo()
