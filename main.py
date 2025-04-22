import torch
import matplotlib.pyplot as plt
from train_ppoM import train_ppo
from train_ppo_attn import train_ppoattn

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter lists
gamma_list = [0.999, 0.9, 0.99] 
mini_batch_list = [32, 64, 128]
gae_lambda_list = [0.8, 0.85, 0.9, 0.99]
clip_ratio_list = [0.2, 0.3, 0.02]
T_list = [2048, 4096]

# Plotting function
def plot_results(results, labels, title):
    plt.figure(figsize=(10, 6))
    for result, label in zip(results, labels):
        plt.plot(result, label=label)

    plt.title(f"Effect of {title}")
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend(title=title)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    filename = f"plots/{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

    plt.show()



# Main experiment loop
def main():
    results = []

    # PPO-M
    # Test clip ratio
    results = []
    print("Testing different clip ratios...")
    for clip_ratio in clip_ratio_list:
        ep_returns = train_ppo(clip_ratio=clip_ratio)
        results.append(ep_returns)
    plot_results(results, [f"clip_ratio={cr}" for cr in clip_ratio_list], "PPOM Clip Ratio")

    # Test buffer size
    results = []
    print("Testing different buffer sizes...")
    for T in T_list:
        ep_returns = train_ppo(T = T)
        results.append(ep_returns)
    plot_results(results, [f"buffer_size={T}" for T in T_list], "PPOM Buffer Size")
    
    # Test gamma
    print("Testing different gamma values...")
    for gamma in gamma_list:
        ep_returns = train_ppo(gamma=gamma)
        results.append(ep_returns)
    plot_results(results, [f"gamma={g}" for g in gamma_list], "PPOM Gamma")
    
    # Test mini-batch size
    results = []
    print("Testing different mini-batch sizes...")
    for mini_batch in mini_batch_list:
        ep_returns = train_ppo(mini_batch_size=mini_batch)
        results.append(ep_returns)
    plot_results(results, [f"mini_batch={mb}" for mb in mini_batch_list], "PPOM Mini Batch Size")
    
    # Test GAE lambda
    results = []
    print("Testing different GAE lambda values...")
    for gae_lambda in gae_lambda_list:
        ep_returns = train_ppo(gae_lambda=gae_lambda)
        results.append(ep_returns)
    plot_results(results, [f"gae_lambda={lam}" for lam in gae_lambda_list], "PPOM GAE Lambda")


    # PPO-attn
    # Test clip ratio
    results = []
    print("Testing different clip ratios...")
    for clip_ratio in clip_ratio_list:
        ep_returns = train_ppoattn(clip_ratio=clip_ratio)
        results.append(ep_returns)
    plot_results(results, [f"clip_ratio={cr}" for cr in clip_ratio_list], "PPOattn Clip Ratio")

    # Test buffer size
    results = []
    print("Testing different buffer sizes...")
    for T in T_list:
        ep_returns = train_ppoattn(T = T)
        results.append(ep_returns)
    plot_results(results, [f"buffer_size={T}" for T in T_list], "PPOattn Buffer Size")
    
    # Test gamma
    print("Testing different gamma values...")
    for gamma in gamma_list:
        ep_returns = train_ppoattn(gamma=gamma)
        results.append(ep_returns)
    plot_results(results, [f"gamma={g}" for g in gamma_list], "PPOattn Gamma")
    
    # Test mini-batch size
    results = []
    print("Testing different mini-batch sizes...")
    for mini_batch in mini_batch_list:
        ep_returns = train_ppoattn(mini_batch_size=mini_batch)
        results.append(ep_returns)
    plot_results(results, [f"mini_batch={mb}" for mb in mini_batch_list], "PPOattn Mini Batch Size")
    
    # Test GAE lambda
    results = []
    print("Testing different GAE lambda values...")
    for gae_lambda in gae_lambda_list:
        ep_returns = train_ppoattn(gae_lambda=gae_lambda)
        results.append(ep_returns)
    plot_results(results, [f"gae_lambda={lam}" for lam in gae_lambda_list], "PPOattn GAE Lambda")
    

# Call main if running directly
if __name__ == "__main__":
    main()
