import os
import sys
import random
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..",".."))

from algorithms.MAA2C.maa2c_agent import MAA2C_Agent
from algorithms.MAA2C.buffer import TrajectoryBuffer
from utils.environment_utils import GeneralizedOvercooked


class MAA2CTrainer:
    """
    Trainer class for MAA2C with a shared policy and  a centralized critic
    """
    def __init__(self, 
                 n_agents:int,
                 maa2c_agent:MAA2C_Agent,
                 environment,
                 layouts:list[str],
                 device:str ="cpu",
                 episode_per_update:int = 10,
                 discount_factor:float = 0.95,
                 gae_decay:float = 0.99,
                 ):
        self.n_agents =n_agents
        self.maa2c_agent = maa2c_agent
        self.environment =environment
        self.layouts = layouts
        self.device = device
        self.episode_per_update =episode_per_update
        self.discount_factor = discount_factor
        self.gae_decay =gae_decay
        self.buffer=TrajectoryBuffer()
        self.current_episode=0
        self.total_rewards=[]
        self.mean_reward=0

    def collect_batch_trajectories(self):
        """
        Function to collect a batch of trajectories by running the agent in the kitchen
        """
        self.buffer.reset()
        batch_rewards=[]

        for _ in range(self.episode_per_update):

            observation=self.environment.reset()
            done,episode_reward=False,0
            self.current_episode += 1

            while not done:
                # we take the observation of both
                state= observation["both_agent_obs"]
                # we select the action
                actions,log_probabilities =self.maa2c_agent.choose_action(state)
                # action performed
                next_observation,reward,done,information = self.environment.step(actions.tolist())
                # we take the reward fot the agents
                shaped_reward = torch.FloatTensor(information["shaped_r_by_agent"]).to(self.device)
                episode_reward +=reward
                agent_rewards=reward+shaped_reward
                # we store the transition
                self.buffer.store_transition(state,actions,log_probabilities,agent_rewards, done)
                observation =next_observation
            batch_rewards.append(episode_reward)

        self.total_rewards.extend(batch_rewards)
        self.mean_reward = np.mean(batch_rewards)

    def advantage_estimation(self,rewards,values,terminated_mask):
        """
        Function to compute the GAE, we first compute the expected return then the gae
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
    
    def training(self, model_name, batches):
        """
        Function to train the MAA2C agent
        """

        actor_losses, critic_losses, mean_rewards = [], [], []
        episode_times=[]

        for batch in tqdm(range(batches),desc="Training Progress",unit="episode"):
            start_time=time.time()
            # Collect trajectories for one batch
            self.collect_batch_trajectories()
            # we convert the buffer to tensors for processing
            states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
            actions = torch.stack(self.buffer.actions).to(self.device)
            rewards = torch.stack(self.buffer.rewards).to(self.device)
            dones = torch.FloatTensor(self.buffer.dones).to(self.device)

            # We flatten the actions for the processing
            actions_flat = actions.view(-1)

            # flatten the states for the critic input
            flatten_states = states.view(states.shape[0],-1)

            # then compute the value estimates using the centralized critic
            state_values = self.maa2c_agent.critic(flatten_states).squeeze(-1)

            # Advantage estimation using GAE
            advantages,expected_returns= self.advantage_estimation(
                rewards,
                state_values,
                dones
            )
            # Expand the advantages to match flattened actions 
            advantages_flat =advantages.unsqueeze(-1).expand(-1, self.n_agents).contiguous().view(-1)

            # Normalize the advantages
            advantages_normalized =(advantages_flat-advantages_flat.mean())/(advantages_flat.std()+ 1e-10)

            # the we flatten states for actor input
            states_flat = states.view(-1,states.shape[-1])

            # now we update policy (MAA2C)
            actor_loss, critic_loss = self.maa2c_agent.update_policy(
                expected_returns,
                states_flat,
                actions_flat,
                advantages_normalized
            )

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            mean_rewards.append(self.mean_reward)

            elapsed = time.time()-start_time
            episode_times.append(elapsed)

            if (batch + 1)%10 == 0:
                self.maa2c_agent.step_schedulers()

            if (batch + 1)%50==0:
                print(f"Batch {batch+1}, Mean reward: {self.mean_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

        self.save_models(model_name)

        return mean_rewards,self.total_rewards,episode_times, actor_losses, critic_losses


    def save_models(self, model_name):
        if model_name:
            base =f"{model_name}"
        elif len(self.layouts)>1:
            base = "maa2c_generalized"
        else:
            base =f"maa2c_{self.env.layout_name}"
        self.maa2c_agent.save_model("actor",f"{base}_actor.pth")
        self.maa2c_agent.save_model("critic", f"{base}_critic.pth")
        print(f"Models saved as {base}_actor.pth and {base}_critic.pth")