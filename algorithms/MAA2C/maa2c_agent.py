import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..",".."))

from algorithms.MAA2C.maa2c_networks import ActorNetwork, CriticNetwork


class MAA2C_Agent:
    """
    Multi-Agent Advantage Actor-Critic agent (maa2c)
    """
    def __init__(
            self,
            n_agents,
            input_dimension,
            action_dimension,
            device:str = "cpu",
            actor_lr:float = 1e-3,
            critic_lr:float = 2e-3,
            scheduler_step_size:int = 1000,
            scheduler_gamma:float = 0.9
    ):
        self.n_agents=n_agents
        self.input_dimension= input_dimension
        self.action_dimension=action_dimension
        self.device=device
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr
        self.scheduler_step_size=scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.device= torch.device(device)
        self.actor = ActorNetwork(input_dimension,action_dimension).to(self.device)
        self.critic =CriticNetwork(2*input_dimension).to(self.device)
        self.mse_loss = nn.MSELoss()
        self.actor_optimizer =self._init_optimizer(self.actor,self.actor_lr)
        self.critic_optimizer = self._init_optimizer(self.critic, self.critic_lr)
        # we try to use learning rate schedulers for adaptive learning (with the episode should stabilize learning)
        self.actor_scheduler = self._init_scheduler(self.actor_optimizer,
                                                   step_size=self.scheduler_step_size,
                                                   gamma=self.scheduler_gamma)
        self.critic_scheduler =self._init_scheduler(self.critic_optimizer,
                                                   step_size=self.scheduler_step_size,
                                                   gamma=self.scheduler_gamma)

    def choose_action(self,obs_n):
        """
        Function to choose the action using the actor Network given the various observations from the kitchen
        """
        obs_n =torch.as_tensor(np.array(obs_n),dtype=torch.float32,device=self.device)
        dist = Categorical(logits=self.actor(obs_n))
        actions =dist.sample()
        log_probability = dist.log_prob(actions)
        return actions,log_probability
    
    def get_value(self,observations):
        """
        Get value estimates from the centralized critic
        """
        if len(observations.shape)==2: # the observation contains the batch size, the number of the agents and the dimension of the obs.
            batch_size = observations.shape[0]
            # For the centralized critic, we need to concatenate observations from both agents
            # we reshape to batch_size/n_agents,n_agents*obs_dim
            reshaped_obs = observations.view(batch_size//self.n_agents,-1)
        else:
            reshaped_obs = observations.view(1, -1)
        return self.critic(reshaped_obs)
    
    def compute_advantages(self,rewards,values,dones,next_values,gamma=0.99,lambda_=0.95):
        """
        In this function we compute the GAE
        """
        advantages =torch.zeros_like(rewards)
        returns =torch.zeros_like(rewards)
        gae=0

        for t in reversed(range(len(rewards))):
            if t == len(rewards)-1:
                next_value=next_values
            else:
                next_value=values[t+1]
            
            delta = rewards[t] +gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma*lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t]+values[t]
        return advantages,returns

    def step_schedulers(self):
        """
        Function to step the lr schedulers
        """
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def update_policy(
        self,
        target_values,
        agent_observations,
        taken_actions,
        advantages):
        """
        Key function to update the policy that guides learning
        """
        # Clear gradients
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        
        batch_size=agent_observations.shape[0]
        # We reshape the observations for centralized critic
        critic_input = agent_observations.view(batch_size//self.n_agents,-1)
        value_estimates = self.critic(critic_input).squeeze()
        
        if target_values.shape!=value_estimates.shape:
            target_values = target_values.view(-1)
        
        critic_loss = self.mse_loss(value_estimates,target_values.detach())
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # actor update
        self.actor_optimizer.zero_grad()
        # we now get the action probabilities for each agent
        dist = Categorical(logits=self.actor(agent_observations))
        
        if len(taken_actions.shape)>1:
            taken_actions=taken_actions.view(-1)
        
        new_log_probs=dist.log_prob(taken_actions)
        entropy=dist.entropy().mean()
        
        if advantages.shape != new_log_probs.shape:
            advantages = advantages.view(-1)

        actor_loss = -(new_log_probs * advantages.detach()).mean() - 0.01 * entropy
        
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(),critic_loss.item()

    def save_model(self,model_type:str,filename:str):
        path =os.path.join(self._get_models_dir(),filename)
        model = getattr(self,model_type)
        torch.save(model.state_dict(),path)

    def load_model(self,model_type:str,filename:str):
        path = os.path.join(self._get_models_dir(),filename)
        model =getattr(self,model_type)
        model.load_state_dict(torch.load(path,map_location=self.device))

    @staticmethod
    def list_saved_models():
        models_dir=MAA2C_Agent._get_models_dir()
        if not os.path.exists(models_dir):
            return []
        return sorted(f for f in os.listdir(models_dir) if f.endswith(".pth"))

    @staticmethod
    def _init_optimizer(network,lr):
        return optim.Adam(network.parameters(),lr=lr)

    @staticmethod
    def _init_scheduler(optimizer, step_size,gamma):
        return optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)

    @staticmethod
    def _get_models_dir():
        current_dir =os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        os.makedirs(models_dir,exist_ok=True)
        return models_dir