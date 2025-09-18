import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image
import sys
import numpy as np
import random
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..",".."))

from algorithms.MAA2C.maa2c_agent import MAA2C_Agent
from algorithms.MAA2C.maa2c_trainer import MAA2CTrainer
from algorithms.MAA2C.buffer import TrajectoryBuffer
from utils.environment_utils import GeneralizedOvercooked
from algorithms.MAPPO.mappo_agent import MAPPO_Agent
from algorithms.MAPPO.mappo_trainer import MAPPOTrainer

def detect_plateau(mean_rewards,episode_times,episodes_per_batch=10,window=5):
    """
    Detect the maximum reward in mean_rewards using moving average to smooth fluctuations.
    """
    rewards =np.array(mean_rewards)
    # moving average
    ma_rewards = np.convolve(rewards, np.ones(window)/window,mode='valid')

    max_idx =np.argmax(ma_rewards)
    max_reward =ma_rewards[max_idx]

    max_episode =max_idx*episodes_per_batch
    max_time = sum(episode_times[:max_idx+window-1])

    return max_reward,max_episode,max_time



def plot_marl_comparison(
    layout,
    mean_rewards_mappo,mean_rewards_maa2c,
    episodes_per_batch_mappo,episodes_per_batch_maa2c,
    mappo_ep,maa2c_ep,
    mappo_max,maa2c_max,
    mappo_time,maa2c_time
):
    """
    We plot the training curves and performance comparison for MAPPO vs MAA2C on a single layout
    """

    sns.set(style="whitegrid",palette="muted",font_scale=1.2)

    plt.figure(figsize=(10,5))
    plt.plot(np.arange(1,len(mean_rewards_mappo)+1)*episodes_per_batch_mappo, mean_rewards_mappo, label="MAPPO",lw=2)
    plt.plot(np.arange(1,len(mean_rewards_maa2c)+1) *episodes_per_batch_maa2c,mean_rewards_maa2c, label="MAA2C", lw=2)
    plt.axvline(mappo_ep,color="blue",ls="--",alpha=0.7)
    plt.axvline(maa2c_ep,color="orange", ls="--",alpha=0.7)
    plt.title(f"Training Rewards - {layout}")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.show()

    print(f"\nPerformance Comparison - {layout}")
    print(f"MAPPO:  Max reward = {mappo_max:.2f}, Episodes to  max = {mappo_ep}, Time to Max = {mappo_time:.2f}s")
    print(f"MAA2C:  Max reward = {maa2c_max:.2f}, Episodes to  max = {maa2c_ep}, Time to Max = {maa2c_time:.2f}s")


def test_soups_delivered(layouts, episodes, agent, actor_path, critic_path):
    """
    Test the generalized agent on each layout individually.
    Creates a dedicated environment for each layout to ensure proper testing.
    """
    
    rewards, soups = [],[]
    agent.load_model("actor",actor_path)
    agent.load_model("critic",critic_path)
    
    for layout in layouts:
        print(f"Testing layout: {layout}")
        
        test_env = GeneralizedOvercooked(layouts=[layout],randomize=False)
        
        rewards_layout, soups_layout =[],[]
        
        for episode in range(episodes):
            obs =test_env.reset()
            
            done =False
            total_reward =0
            
            while not done:
                agent_observations =obs["both_agent_obs"]
                actions =[]
                
                for agent_id in range(2):
                    action, _ = agent.choose_action(agent_observations[agent_id])
                    actions.append(action.item() if hasattr(action,'item') else int(action))
                
                # Step
                obs,reward,done,info = test_env.step(actions)
                
                total_reward +=reward
            
            # at the end soups delivered
            soup_deliveries_agent0 =len(info['episode']['ep_game_stats']['soup_delivery'][0])
            soup_deliveries_agent1 = len(info['episode']['ep_game_stats']['soup_delivery'][1])
            total_soups = soup_deliveries_agent0+ soup_deliveries_agent1
            print(total_soups)
            rewards_layout.append(total_reward)
            soups_layout.append(total_soups)
        
        print(f" Completed {layout}: {episodes} episodes")
        rewards.append(rewards_layout)
        soups.append(soups_layout)
    
    return rewards, soups


def evaluate_agent(layouts,test_episodes, agent, actor_path,critic_path):
    """
    we evaluate an agent across multiple layouts and print average rewards and soups delivered
    """
    rewards, soups = test_soups_delivered(
        layouts=layouts,
        episodes=test_episodes,
        agent=agent,
        actor_path=actor_path,
        critic_path=critic_path
    )

    print(f"\nTest Results over {test_episodes} episodes per layout:")
    for i,layout in enumerate(layouts):
        avg_reward= sum(rewards[i]) /len(rewards[i]) if  rewards[i] else 0.0
        avg_soups= sum(soups[i]) /len(soups[i]) if  soups[i] else 0.0

        print(f"  Layout: {layout}")
        print(f"    Avg reward: {avg_reward:.2f}" )
        print(f"    Avg soups Delivered: {avg_soups:.2f}")

    return rewards, soups

def train_and_plot_mappo(layouts, model_name,iters,episodes_per_batch,batches):
    """
    function to train a MAPPO agent on given layouts and plot the training rewards
    """
    print(f"Training on multiple layouts: {layouts}")

    env = GeneralizedOvercooked(layouts=layouts,randomize=True)

    input_dim =env.observation_space.shape[-1]
    action_dim =env.action_space.n

    mappo_agent = MAPPO_Agent(
        input_dimension=input_dim,
        action_dimension=action_dim,
        iters=iters
    )

    mappo_trainer = MAPPOTrainer(
        n_agents=2,
        mappo_agent=mappo_agent,
        episode_per_update=episodes_per_batch,
        environment=env,
        layouts=layouts
    )

    mean_rewards,episode_rewards,episode_time,actor_losses, critic_losses =mappo_trainer.training(
        model_name=model_name,
        batches=batches,
    )

    sns.set(style="whitegrid",palette="muted",font_scale=1.2)
    plt.figure(figsize=(12,6))
    episodes =np.arange(1,len(mean_rewards)+1)*episodes_per_batch
    plt.plot(episodes,mean_rewards,label="generalized",color="red")
    plt.axhline(50,color="purple",ls="--",alpha=0.7)
    plt.title(f"MAPPO mean rewards on {', '.join(layouts)}")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.legend(title="Layout")
    plt.tight_layout()
    plt.show()

    return mappo_agent,mean_rewards,episode_rewards, episode_time, actor_losses,critic_losses


def train_and_plot_maa2c(layouts,model_name, episodes_per_batch,batches,n_agents=2):
    """
    Function to train MAA2C agent on given layouts and plot the training rewards
    """
    print(f"Training on multiple layouts: {layouts}")

    env =GeneralizedOvercooked(layouts=layouts,randomize=True)

    input_dim =env.observation_space.shape[-1]
    action_dim =env.action_space.n

    maa2c_agent =MAA2C_Agent(
        input_dimension=input_dim,
        action_dimension=action_dim,
        n_agents=n_agents
    )

    maa2c_trainer =MAA2CTrainer(
        n_agents=n_agents,
        maa2c_agent=maa2c_agent,
        episode_per_update=episodes_per_batch,
        environment=env,
        layouts=layouts
    )

    mean_rewards,episode_rewards,episode_time,actor_losses, critic_losses= maa2c_trainer.training(
        model_name=model_name,
        batches=batches,
    )

    # Plot results
    sns.set(style="whitegrid",palette="muted",font_scale=1.2 )
    plt.figure(figsize=(12,6))
    episodes =np.arange(1,len(mean_rewards)+1)*episodes_per_batch
    plt.plot(episodes,mean_rewards,label="generalized",color="red")
    plt.axhline(50, color="purple",ls="--",alpha=0.7 )
    plt.title(f"MAA2C Mean Rewards on {', '.join(layouts)}")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.legend(title="Layout")
    plt.tight_layout()
    plt.show()

    return maa2c_agent,mean_rewards, episode_rewards, episode_time,actor_losses, critic_losses

def cross_algorithm(env_class,layouts,agent1, agent2,agent1_actor_path,agent2_actor_path,n_episodes=5,randomize_positions=True):
    """
    function to evaluate the combination of two different agents in a shared environment.
    """
    env = env_class(layouts=layouts,randomize=True)
    input_dim =env.observation_space.shape[-1]
    action_dim =env.action_space.n

    agent1.load_model("actor",agent1_actor_path)
    agent2.load_model("actor",agent2_actor_path)

    episode_rewards = []

    for ep in range(n_episodes):
        obs =env.reset()
        done,ep_reward =False,0

        while not done:
            actions =[]
            obs_n =obs["both_agent_obs"]

            # optionally werandomize agent positions
            if randomize_positions:
                agents = [(agent1,obs_n[0]),(agent2,obs_n[1])]
                random.shuffle(agents)
                a0, _ =agents[0][0].choose_action([agents[0][1]])
                a1, _ =agents[1][0].choose_action([agents[1][1]])
            else:
                a0, _ =agent1.choose_action([obs_n[0]])
                a1, _ =agent2.choose_action([obs_n[1]])

            actions = [a0.item(),a1.item()]

            obs, reward, done,info= env.step(actions)
            ep_reward+=reward

        episode_rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward = {ep_reward:.2f}")

    avg_reward =np.mean(episode_rewards)
    print(f"\nAverage reward over {n_episodes} episodes: {avg_reward:.2f}")
    return episode_rewards

def run_cross_algorithm(layouts,agent1_path,agent2_path,n_episodes=5,randomize_positions=True):
    """helper funct. to run cross-algorithm experiments on given layouts"""
    env =GeneralizedOvercooked(layouts=layouts,randomize=True)
    input_dim =env.observation_space.shape[-1]
    action_dim = env.action_space.n

    mappo_agent = MAPPO_Agent(input_dimension=input_dim,action_dimension=action_dim)
    maa2c_agent = MAA2C_Agent(n_agents=2,input_dimension=input_dim,action_dimension=action_dim)

    return cross_algorithm(
        env_class=GeneralizedOvercooked,
        layouts=layouts,
        agent1=maa2c_agent,
        agent2=mappo_agent,
        agent1_actor_path=agent1_path,
        agent2_actor_path=agent2_path,
        n_episodes=n_episodes,
        randomize_positions=randomize_positions
    )
