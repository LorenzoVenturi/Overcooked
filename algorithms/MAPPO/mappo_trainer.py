import os
import sys
import random
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..",".."))

from algorithms.MAPPO.mappo_agent import MAPPO_Agent
from algorithms.MAPPO.buffer import TrajectoryBuffer
from utils.environment_utils import GeneralizedOvercooked


class MAPPOTrainer:
    """
    Trainer class for Multi-Agent PPO with a shared policy and a centralized critic
    """
    def __init__(self, 
                 n_agents:int,
                 mappo_agent: MAPPO_Agent,
                 environment,
                 layouts:list[str],
                 device:str ="cpu",
                 episode_per_update:int = 10,
                 discount_factor:float = 0.95,
                 gae_decay:float = 0.99,
                 ):
        
        self.n_agents= n_agents
        self.mappo_agent= mappo_agent
        self.environment= environment
        self.layouts= layouts
        self.device= torch.device(device)

        self.episode_per_update = episode_per_update
        self.discount_factor = discount_factor
        self.gae_decay =gae_decay

        self.current_episode=0
        self.total_rewards=[]
        self.mean_reward=0
        
        self.buffer=TrajectoryBuffer()

    def collect_batch_trajectories(self):
        """
        Function to collect a batch of trajectories by running the agent in the environment
        """
        self.buffer.reset()
        batch_rewards =[]

        for _ in range(self.episode_per_update):

            observation=self.environment.reset()
            done,episode_reward=False,0
            self.current_episode += 1

            while not done:

                state= observation["both_agent_obs"]
                
                actions,log_probabilities=self.mappo_agent.choose_action(state)

                next_observation,reward,done,information= self.environment.step(actions.tolist())

                shaped_reward = torch.FloatTensor(information["shaped_r_by_agent"]).to(self.device)
                episode_reward += reward
                agent_rewards = reward+shaped_reward
                self.buffer.store_transition(state,actions,log_probabilities,agent_rewards,done)
                observation = next_observation
            
            batch_rewards.append(episode_reward)

        self.total_rewards.extend(batch_rewards)
        self.mean_reward=np.mean(batch_rewards)

    def advantage_estimation(self, rewards, values, terminated_mask):
        """
        Function to compute the gae
        """
        advantages,returns,gae = [],[],0
        
        
        total_rewards = rewards.sum(dim=1) if rewards.dim() > 1 else rewards
        
        for step in reversed(range(len(total_rewards))):
            next_value=0 if step==len(total_rewards)-1 else values[step+1].detach()*(1-terminated_mask[step])
            expected_return = total_rewards[step] + self.discount_factor*next_value
            delta= expected_return - values[step].detach()
            gae= delta + self.discount_factor*self.gae_decay*gae
            advantages.insert(0,gae)
            returns.insert(0,expected_return)
            
        return torch.stack(advantages).to(self.device), torch.stack(returns).to(self.device)

    def training(self, model_name, batches, pretrained_path_actor=None, pretrained_path_critic=None):
        """
        Function to train the MAPPO agent
        """

        actor_losses,critic_losses,mean_rewards = [],[],[]

        episode_times=[]
        # Load pre-trained model if path is provided
        if pretrained_path_actor is not None:
            self.mappo_agent.load_model("actor",pretrained_path_actor)
            print(f"Loaded pre-trained actor model from {pretrained_path_actor}")

        if pretrained_path_critic is not None:
            self.mappo_agent.load_model("critic",pretrained_path_critic)
            print(f"Loaded pre-trained critic model from {pretrained_path_critic}")

        for batch in tqdm(range(batches),desc="Training Progress",unit="episode"):
            start_time=time.time()
            self.collect_batch_trajectories()

            states=torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
            actions = torch.stack(self.buffer.actions).to(self.device)
            log_probabilities = torch.stack(self.buffer.log_probs).to(self.device)
            rewards = torch.stack(self.buffer.rewards).to(self.device)
            dones = torch.FloatTensor(self.buffer.dones).to(self.device)

            # we flatten the action and the log porbs.
            actions_flat = actions.view(-1)
            log_probs_flat = log_probabilities.view(-1)

            # we flatten all the state of all agents into a joint representation
            flatten_states= states.view(states.shape[0],-1)
            
            # then we compute the value estimate ( the output critic) and remove the extra dimension
            state_values = self.mappo_agent.critic(flatten_states).squeeze(-1)
            
            advantages,expected_returns= self.advantage_estimation(
                rewards,
                state_values,
                dones
            )

            # Expand advantages to match flattened actions (each timestep advantage applies to both agents)
            advantages_flat=advantages.unsqueeze(-1).expand(-1,self.n_agents).contiguous().view(-1)
            
            # Normalize advantages
            advantages_normalized=(advantages_flat - advantages_flat.mean()) / (advantages_flat.std()+1e-10)

            actor_loss,critic_loss =self.mappo_agent.update_policy(
                expected_returns,
                states,
                actions_flat,
                log_probs_flat,
                advantages_normalized
            )

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            mean_rewards.append(self.mean_reward)
            
            elapsed = time.time() - start_time
            episode_times.append(elapsed)

            # Step learning rate schedulers
            if (batch+1) % 10 == 0:
                self.mappo_agent.step_schedulers()

            if (batch+1) % 50 == 0:
                print(f"Batch {batch+1}, Mean Reward: {self.mean_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

        self.save_models(model_name)

        return mean_rewards, self.total_rewards, episode_times, actor_losses, critic_losses

    def save_models(self, model_name):
        if model_name:
            base =f"{model_name}"
        elif len(self.layouts) > 1:
            base = "mappo_generalized"
        else:
            base =f"mappo_{self.env.layout_name}"

        # we save both actor and critic models
        self.mappo_agent.save_model("actor", f"{base}_actor.pth")
        self.mappo_agent.save_model("critic", f"{base}_critic.pth")
        print(f"Models saved as {base}_actor.pth and {base}_critic.pth")