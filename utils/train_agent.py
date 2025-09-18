import sys
import os
from pathlib import Path
import argparse
import torch

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
overcooked_src = project_root / "overcooked_ai" / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src)) 

# Import after path setup
from utils.training_utils import train_and_plot_mappo, train_and_plot_maa2c

def main():
    parser = argparse.ArgumentParser(description="train MARL agents for Overcooked")
    
    # required arguments:
    parser.add_argument('--algorithm','-a',choices=['MAA2C','MAPPO'],
                        help='Algorithm to train')
    parser.add_argument('--model-name','-m',type=str,
                        help='Name for the trained model (e.g. my_custom_model)')
    
    # layout arguments:
    parser.add_argument('--layout','-l',type=str,action='append',
                        choices=['cramped_room','coordination_ring','forced_coordination','asymmetric_advantages'],
                        help='Layout(s) to train on (can be used multiple times for multi-layout training)')
    parser.add_argument('--all-layouts',action='store_true',
                        help='Train on all available layouts')
    
    # training parameters:
    parser.add_argument('--batches', '-b', type=int,default=100,
                        help='Number of training batches (default: 100)')
    parser.add_argument('--episodes-per-batch', '-e',type=int,default=10,
                        help='Episodes per training batch (default: 10)')
    parser.add_argument('--iters', '-i', type=int, default=10,
                        help='number of iterations for MAPPO agent (default: 10)')
    
    # device
    parser.add_argument('--device','-d',choices=['cpu','cuda'],default='cpu',
                        help='Device to use for training (default:cpu)')
    
    # display help
    parser.add_argument('--list-layouts',action='store_true',
                        help='list all the available layouts')
    
    args = parser.parse_args()
    
    if args.list_layouts:
        print("Available layouts:")
        print("  - cramped_room")
        print("  - coordination_ring") 
        print("  - forced_coordination")
        print("  - asymmetric_advantages")
        return
    
    if not args.algorithm:
        parser.error("--algorithm/-a is required for training")
    if not args.model_name:
        parser.error("--model-name/-m is required for training")
    
    if not args.layout and not args.all_layouts:
        parser.error("You must specify at least one layout using --layout or use --all-layouts")
    
    if args.all_layouts:
        layouts = ['cramped_room','coordination_ring','forced_coordination','asymmetric_advantages']
    else:
        layouts =args.layout
    
    if args.device=='cuda' and not torch.cuda.is_available():
        print("CUDA not available-> using CPU instead")
        device ='cpu'
    else:
        device= args.device
    
    print("starting the training")
    print(f"algorithm: {args.algorithm}" )
    print(f"model Name: {args.model_name}" )
    print(f"layouts: {layouts}")
    print(f"batches: {args.batches}")
    print(f"episodes per batch: {args.episodes_per_batch}")
    if args.algorithm=='MAPPO':
        print(f"iterations: {args.iters}")
    print(f"device: {device}")
    print() 
    
    # training
    
    if args.algorithm=='MAPPO':
        print("Training the MAPPO agent")
        agent, mean_rewards, episode_rewards, episode_times,actor_losses,critic_losses =train_and_plot_mappo(
            layouts=layouts,
            model_name=args.model_name,
            iters=args.iters,
            episodes_per_batch=args.episodes_per_batch,
            batches=args.batches
        )
    
    elif args.algorithm == 'MAA2C':
        print("training the MAA2C agent")
        agent,mean_rewards,episode_rewards,episode_times,actor_losses,critic_losses= train_and_plot_maa2c(
            layouts=layouts,
            model_name=args.model_name,
            episodes_per_batch=args.episodes_per_batch,
            batches=args.batches
        )
    
    # we display the final training results
    print("\n-- training results --")
    print(f"final mean reward: {mean_rewards[-1]:.2f}" )
    print(f"best mean reward: {max(mean_rewards):.2f}" )
    print(f"total episodes: {len(episode_rewards)}"  )
    print(f"final actor loss: {actor_losses[-1]:.4f}" )
    print(f"final critic loss: {critic_losses[-1]:.4f}" )
    
    # saving
    models_dir=project_root / "algorithms" / args.algorithm / "models"
    print(f"\ntraining completed. models saved in: {models_dir}")
    print(f"actor model: {args.model_name}_actor.pth")
    print(f"critic model: {args.model_name}_critic.pth")
    
    print(f"\nto test your trained model:")
    print(f"python utils/simple_render.py --algorithm {args.algorithm} --model {args.model_name} --layout {layouts[0]}")
    

if __name__ == "__main__":
    main()