import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..",".."))

from algorithms.MAPPO.networks import ActorNetwork, CriticNetwork

class MAPPO_Agent:
    """
    Multi-Agent Proximal Policy Optimization agent
    """
    def __init__(
            self,
            input_dimension,
            action_dimension,
            device:str = "cpu",
            clip_range:float =0.2,
            actor_lr:float =3e-4,
            critic_lr:float =1e-3,
            entropy: float = 0.01,
            iters:int = 10,
            step_size:int = 1000,
            gamma:float = 0.9,
            is_training:bool = True,
            norm:bool = False
    ):
        self.device= torch.device(device)
        self.actor = ActorNetwork(input_dimension,action_dimension).to(self.device)

        self.clip_range =clip_range
        self.entropy_coef =entropy
        self.iters =iters
        self.mse_loss =nn.MSELoss()

        self.step_size = step_size
        self.gamma = gamma
        self.norm=norm

        if is_training:
            self.critic = CriticNetwork(2*input_dimension).to(self.device)
            self.actor_optimizer = self._init_optimizer(self.actor, actor_lr)
            self.critic_optimizer = self._init_optimizer(self.critic, critic_lr)

            # we try to use learning rate schedulers for adaptive learning
            self.actor_scheduler = self._init_scheduler(self.actor_optimizer, step_size=self.step_size, gamma=self.gamma)
            self.critic_scheduler = self._init_scheduler(self.critic_optimizer, step_size=self.step_size, gamma=self.gamma)

    @torch.no_grad()
    def choose_action(self, obs_n):
        """
        Function to choose the action using the actor Network given the various observations
        """
        obs_n = torch.as_tensor(np.array(obs_n), dtype=torch.float32, device=self.device)
        dist = Categorical(logits=self.actor(obs_n))
        actions = dist.sample()
        log_probability = dist.log_prob(actions)

        return actions,log_probability
        

    def update_policy(
        self,
        target_values,
        agent_observations,
        taken_actions,
        previous_log_probs,
        advantages,
        ):
        actor_loss, critic_loss = None, None

        # Flattened joint states for the critic
        batch_size= agent_observations.shape[0]
        num_agents= agent_observations.shape[1]
        state_dim= agent_observations.shape[2]

        for _ in range(self.iters):
            
            joint_states= agent_observations.reshape(batch_size,-1)
            state_values= self.critic(joint_states).squeeze(-1)

            
            per_agent_states = agent_observations.reshape(batch_size*num_agents,state_dim)
            logits = self.actor(per_agent_states)
            dist = Categorical(logits=logits)

            
            new_log_probs= dist.log_prob(taken_actions)
            entropy= dist.entropy().mean()

            
            ratios= torch.exp(new_log_probs - previous_log_probs)
            unclipped_loss = ratios * advantages
            clipped_loss = torch.clamp(ratios,1-self.clip_range,1+self.clip_range)*advantages
            policy_loss = torch.min(unclipped_loss, clipped_loss).mean()

            
            actor_loss= -(policy_loss-self.entropy_coef*entropy)
            critic_loss= self.mse_loss(state_values, target_values)

            
            self._optimize(self.actor_optimizer,actor_loss,retain_graph=True,norm=self.norm)
            self._optimize(self.critic_optimizer, critic_loss,norm=self.norm)

        return actor_loss.item(), critic_loss.item()

    def step_schedulers(self):
        """
        Step the learning rate schedulers
        """
        if hasattr(self,'actor_scheduler'):
            self.actor_scheduler.step()
        if hasattr(self,'critic_scheduler'):
            self.critic_scheduler.step()
    
    def save_model(self,model_type:str,filename:str):
        path = os.path.join(self._get_models_dir(),filename)
        model = getattr(self,model_type)
        torch.save(model.state_dict(), path)

    def load_model(self, model_type: str, filename: str):
        path = os.path.join(self._get_models_dir(),filename)
        model = getattr(self, model_type)
        model.load_state_dict(torch.load(path, map_location=self.device))


    @staticmethod
    def list_saved_models():
        models_dir =MAPPO_Agent._get_models_dir()
        if not os.path.exists(models_dir):
            return []
        return sorted(f for f in os.listdir(models_dir) if f.endswith(".pth"))
    
    @staticmethod
    def _init_optimizer(model,lr):
        return optim.Adam(model.parameters(),lr=lr)

    @staticmethod
    def _init_scheduler(optimizer,step_size=1000,gamma=0.9):
        return optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
    
    @staticmethod
    def _optimize(optimizer, loss, retain_graph: bool = False,norm:bool=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if norm:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=0.5)
        optimizer.step()

    @staticmethod
    def _get_models_dir():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        return models_dir