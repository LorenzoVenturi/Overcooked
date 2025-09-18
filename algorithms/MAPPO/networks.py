import torch.nn as nn

class ActorNetwork(nn.Module):
    """
    Actor network for the policy training
    """
    def __init__(self,input_dim,action_dim,hidden_sizes=[128,256,128]):
        super().__init__()
        self.input_n = nn.LayerNorm(input_dim)

        # block of layers
        layers= []
        prev_size= input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size,hidden_size),
                nn.ReLU()
            ])
            prev_size=hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_size, action_dim)
        
    def forward(self,state):
        x = self.input_n(state)
        x = self.hidden_layers(x)
        logits = self.policy_head(x)
        return logits


class CriticNetwork(nn.Module):
    """
     Critic network for value function approx.
    """
    def __init__(self, input_dim, hidden_sizes=[128,256,128]):
        super().__init__()
        self.input_n = nn.LayerNorm(input_dim)

        # block of layers
        layers =[]
        prev_size =input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size,hidden_size),
                nn.ReLU()
            ])
            prev_size =hidden_size
            
        self.hidden_layers =nn.Sequential(*layers)
        self.value_head =nn.Linear(prev_size, 1)
        
    def forward(self,state):
        x = self.input_n(state)
        x = self.hidden_layers(x)
        value = self.value_head(x)
        return value
    