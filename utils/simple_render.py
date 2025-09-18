import sys
import os
from pathlib import Path
import torch
import numpy as np
import pygame
import argparse

script_dir=Path(__file__).parent.absolute()
project_root=script_dir.parent
overcooked_src=project_root / "overcooked_ai" / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0,str(project_root))

if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))

if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from algorithms.MAA2C.maa2c_agent import MAA2C_Agent
from algorithms.MAPPO.mappo_agent import MAPPO_Agent
from utils.environment_utils import GeneralizedOvercooked

def discover_models(algorithm):
    models={}
    
    if algorithm.upper()=="MAA2C":
        models_dir=project_root / "algorithms" / "MAA2C" / "models"
    elif algorithm.upper()=="MAPPO":
        models_dir = project_root / "algorithms" / "MAPPO" / "models"
    else:
        return {}
    
    if not models_dir.exists():
        return {}
    
    actor_files = list(models_dir.glob("*_actor.pth"))
    
    for actor_file in actor_files:
        model_name =actor_file.stem[:-6] 
        critic_file =models_dir / f"{model_name}_critic.pth"
        if critic_file.exists():
            models[model_name] =(actor_file.name, critic_file.name)
    
    return models

def get_available_models():
    maa2c_models= discover_models("MAA2C")
    mappo_models= discover_models("MAPPO")
    return maa2c_models, mappo_models

# we get the models actually available
MAA2C_MODELS,MAPPO_MODELS = get_available_models()


class Model_Renderer:
    def __init__(self, algorithm, model_name,layout,frame_rate=10, device="cpu"):
        self.algorithm = algorithm.upper()
        self.model_name = model_name
        self.layout = layout
        self.frame_rate =frame_rate
        self.frame_duration=1000 // frame_rate
        self.device =torch.device(device)
        self.env = GeneralizedOvercooked(layouts=[layout],randomize=False)
        obs = self.env.reset()
        input_dim = len(obs["both_agent_obs"][0])
        action_dim = self.env.action_space.n
        
        print(f"Environment: {layout}, Input dim: {input_dim}, Action dim: {action_dim}")
        
        # load the agent based on the specific algorithm
        self.agent = self._load_agent(input_dim, action_dim)
        
        self.visualizer = StateVisualizer()
        
        print(f"Loaded {algorithm} agent: {model_name}")
    
    def _load_agent(self,input_dim,action_dim):
        if self.algorithm=="MAA2C":
            if self.model_name not in MAA2C_MODELS:
                raise ValueError(f"MAA2C model '{self.model_name}' not found. avaivable:{list(MAA2C_MODELS.keys())}")
            agent =MAA2C_Agent(
                n_agents=2,
                input_dimension=input_dim,
                action_dimension=action_dim,
                device=self.device.type
            )
            
            actor_file, critic_file = MAA2C_MODELS[self.model_name]
            agent.load_model("actor",actor_file)
            agent.load_model("critic",critic_file)
            return agent
            
        elif self.algorithm=="MAPPO":
            if self.model_name not in MAPPO_MODELS:
                raise ValueError(f"MAPPO model '{self.model_name}' not found. available : {list(MAPPO_MODELS.keys())}")
                
            agent =MAPPO_Agent(
                input_dimension=input_dim,
                action_dimension=action_dim,
                device=self.device.type,
                is_training=False
            )
            
            actor_file,_=MAPPO_MODELS[self.model_name]
            agent.load_model("actor", actor_file)
            return agent
        else:
            raise ValueError(f"use 'MAA2C' or 'MAPPO'")
    
    def _get_actions(self, obs):
        states = obs["both_agent_obs"]
        states = torch.FloatTensor(np.array(states)).to(self.device)
        with torch.no_grad():
            if self.algorithm=="MAA2C":
                actions, _=self.agent.choose_action(states)
                return [action.item() for action in actions]
            else:
                actions = []
                for state in states:
                    action, _ = self.agent.choose_action(state.unsqueeze(0))
                    actions.append(action.item())
                return actions
    
    def play_episode(self):
        id_o=None
        while id_o != 1:
            obs = self.env.reset()
            id_o = obs["other_agent_env_idx"]
        done=False

        trajectory =[]
        hud =[]

        episode_reward =0
        soup_count =0
        steps =0

        while not done:
            trajectory.append(obs["overcooked_state"])
            hud.append({"score": episode_reward, "soups": soup_count, "steps": steps})
            actions = self._get_actions(obs)
            obs,reward, done, info = self.env.step(actions)
            soup_count += reward // 20
            episode_reward += reward
            steps += 1

        return trajectory, hud, episode_reward, soup_count

    def render_mov(self):
        traj, hud, reward, soups = self.play_episode()
        print(f"Rendering: Layout: {self.layout} - Final Reward: {reward} - Soup Count: {soups}")

        base_mdp = self.env.curr_env.base_env.mdp

        # Generate frames
        frames = [
            self.visualizer.render_state(
                state,grid=base_mdp.terrain_mtx, hud_data=hud_data
            )
            for state,hud_data in zip(traj,hud)
        ]

        if len(frames)==0:
            print("No frames")
            return {
                "reward": reward,
                "soups": soups,
                "layout": self.layout,
                "algorithm": self.algorithm,
                "model_name": self.model_name
            }

        pygame.init()
        first_frame=frames[0]
        window=pygame.display.set_mode(first_frame.get_size())
        pygame.display.set_caption(f"{self.algorithm} - {self.model_name} - {self.layout}")

        clock =pygame.time.Clock()
        frame_count =0
        running= True
        frame= first_frame

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running=False

            window.blit(frame,(0,0))
            pygame.display.flip()

            clock.tick(self.frame_rate)
            # we update the frame
            frame_count+=1
            if frame_count<len(frames):
                frame=frames[frame_count]
            elif frame_count>=len(frames) +int(self.frame_rate*2):
                running=False

        pygame.quit()

        return {
            "reward":reward,
            "soups":soups,
            "layout":self.layout,
            "algorithm": self.algorithm,
            "model_name": self.model_name
        }



def safe_simple_render(algorithm,model_name,layout,num_episodes=1,frame_rate=10,device="cpu"):
    """
    wrapper for the Model_Renderer class structure
    """
    print(f"\n-- Render: {algorithm} - {model_name} on {layout} --")
    
    results =[]
    all_rewards =[]
    all_soup_counts =[]
    
    for episode in range(num_episodes):
        print(f"running episode {episode + 1}/{num_episodes}")
        
        tester =Model_Renderer(
            algorithm=algorithm,
            model_name=model_name,
            layout=layout,
            frame_rate=frame_rate,
            device=device
        )
        
        # we render the episode
        episode_result =tester.render_mov()
        
        results.append(episode_result)
        all_rewards.append(episode_result["reward"])
        all_soup_counts.append(episode_result["soups"])
        
        print(f"Episode {episode + 1}: Reward = {episode_result['reward']:.2f}, Soups = {episode_result['soups']}")
    
    # final results
    avg_reward = np.mean(all_rewards) if all_rewards else 0.0
    avg_soups = np.mean(all_soup_counts) if all_soup_counts else 0.0
    
    final_results = {
        "rewards":all_rewards,
        "avg_reward": avg_reward,
        "soup_counts": all_soup_counts,
        "avg_soups":avg_soups,
        "layout": layout,
        "algorithm":algorithm,
        "model_name": model_name
    }
    print(f"\n-- Results --")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average soups: {avg_soups:.2f}")
    print(f"Rewards per episode: {all_rewards}")
    print(f"Soups per episode: {all_soup_counts}")
    return final_results


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="render the MARL agents playing overcooked with pygame visualization")
    parser.add_argument("--algorithm","-a",choices=["MAA2C","MAPPO"], 
                       help="algorithm to use (MAA2C or MAPPO)")
    parser.add_argument("--model", "-m",
                       help="Model name to load")
    parser.add_argument("--layout","-l",
                       choices=["cramped_room","coordination_ring","forced_coordination", "asymmetric_advantages"],
                       help="layout to render")
    parser.add_argument("--episodes","-e", type=int,default=1,
                       help="number of episodes to render (default: 1)")
    parser.add_argument("--frame-rate", "-f",type=int,default=10,
                       help="frame rate for pygame rendering (default:10)")
    parser.add_argument("--device","-d",default="cpu",choices=["cpu","cuda"],
                       help="device to use for inference (default: cpu)")
    parser.add_argument("--list-models", action="store_true",
                       help="list all available models and exit")

    args = parser.parse_args()
    
    if args.list_models:
        maa2c_models, mappo_models = get_available_models()

        print("\n-- available MAA2C Models --")
        if maa2c_models:
            for model in sorted(maa2c_models.keys()):
                print(f"  - {model}")
        else:
            print("  no MAA2C models found")

        print("\n-- available MAPPO Models --")
        if mappo_models:
            for model in sorted(mappo_models.keys()):
                print(f"  - {model}")
        else:
            print("  no MAPPO models found")

        print("\n-- available Layouts --")
        layouts = ["cramped_room","coordination_ring", "forced_coordination","asymmetric_advantages"]
        for layout in layouts:
            print(f"  - {layout}")
        exit(0)
    
    if not args.algorithm:
        parser.error("the following arguments are required: --algorithm/-a")
    if not args.model:
        parser.error("the following arguments are required: --model/-m")
    if not args.layout:
        parser.error("the following arguments are required: --layout/-l")
    
    
    print(f"Algorithm: {args.algorithm}")
    print(f"Model: {args.model}")
    print(f"Layout: {args.layout}")
    print(f"Episodes: {args.episodes}")
    print(f"Frame Rate: {args.frame_rate}")
    print(f"Device: {args.device}")
    
    results = safe_simple_render(
        algorithm=args.algorithm,
        model_name=args.model,
        layout=args.layout,
        num_episodes=args.episodes,
        frame_rate=args.frame_rate,
        device=args.device
    )
    
    print(" render completed")
