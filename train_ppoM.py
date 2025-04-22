import torch
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from ppoM import Trainer

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_ppo(clip_ratio=0.2, gamma=0.99, gae_lambda=0.8, entropy_coef = 1e-4, mini_batch_size=64, 
            actor_lr=3e-4, critic_lr=4e-4, n_epochs=5, T = 2048, n_episodes=3000, hardcore=False):


    env = gym.make("BipedalWalker-v3", hardcore=hardcore, render_mode=None)
    seed = random.randint(0, 2**32 - 1)
    env = set_seed(seed, env)
    trainer = Trainer(clip_ratio, gamma, gae_lambda, entropy_coef, mini_batch_size, actor_lr, critic_lr, n_epochs)

    #n_episodes = 2000
    episode_rewards = []
    avg_rewards = []
    max_reward = -1000
    steps_at_maxr = 0

    for episode in (pbar := tqdm(range(n_episodes), desc="avg reward = N/A / max reward = N/A ")):
        steps = 0
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0

        # Convert state to tensor and move it to the correct device (CPU or GPU)
        state = torch.tensor(state, dtype=torch.float32).to(device)

        while not terminated and not truncated:
            steps += 1
            if len(state.shape) == 1:  # Check if the state is 1D
                state = state.unsqueeze(0).unsqueeze(0)
            action, prob, value = trainer.agent.choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Convert new state to tensor and move it to the correct device
            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)

            trainer.remember(state, action, reward, value, prob, terminated)
            episode_reward += reward

            if len(trainer.memory.states) >= T:
                trainer.learn()
                trainer.memory.clear_memory()

            state = new_state  # Move the new_state to the next state in the loop

        episode_rewards.append(episode_reward)
        if episode_reward > max_reward:
            max_reward = episode_reward
            steps_at_maxr = steps
            torch.save(trainer.agent.actor.state_dict(), f"models/ppo_actor_gamma:{gamma:.3f}_lambda:{gae_lambda:.3f}_batch:{mini_batch_size}_clip:{clip_ratio:.3f}.pt")
            torch.save(trainer.agent.critic.state_dict(), f"models/ppo_critic_gamma:{gamma:.3f}_lambda:{gae_lambda:.3f}_batch:{mini_batch_size}_clip:{clip_ratio:.3f}.pt")
        avg_reward = np.mean(episode_rewards[-100:])  # Take average of last 100 episodes
        avg_rewards.append(avg_reward)
        pbar.set_description(f"avg reward = {avg_reward:.2f} / max reward = {max_reward:.2f} / Number of steps for max rewards = {steps_at_maxr:.2f} ")


    env.close()


    #to_save = input("Want to save this file? Press N if no\n")
    #if to_save.lower() == 'y':
    #torch.save(trainer.agent.actor.state_dict(), f"models/ppo_actor_clip:{clip_ratio:.2f}_gamma:{gamma:.2f}_lambda:{gae_lambda:.2f}_batch:{mini_batch_size}_clip:{clip_ratio:.2f}.pt")
    #torch.save(trainer.agent.critic.state_dict(), f"models/ppo_critic_clip:{clip_ratio:.2f}_gamma:{gamma:.2f}_lambda:{gae_lambda:.2f}_batch:{mini_batch_size}_clip:{clip_ratio:.2f}.pt")
    return avg_rewards#, max_reward, steps_at_maxr


  