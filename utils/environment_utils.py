from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
import random

def make_env(layout_name,horizon=400,info_level=0):
    """
    Initialize the Overcooked environment"""

    mdp = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=True)
    base_env = OvercookedEnv.from_mdp(mdp=mdp, horizon=horizon, info_level=info_level)

    return Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

class GeneralizedOvercooked:
    """
    A wrapper for multiple Overcooked environments with different layouts
    """
    def __init__(self, layouts,horizon=400,info_level=0,randomize=True):

        if not layouts:
            raise ValueError("We need at least 1 layout")
        
        self.layouts =layouts
        self.randomize =randomize

        self.environments = [make_env(layout, horizon=horizon, info_level=info_level) for layout in layouts]

        self.env_idx = 0
        self.curr_env=self.environments[self.env_idx]

        self.layout_name = self.curr_env.base_env.mdp.layout_name
        self.observation_space = self.curr_env.observation_space
        self.action_space = self.curr_env.action_space

    def reset(self):
        """
        Reset the environment. If randomize is True, randomly select an environment from the list.
        """
        if self.randomize and len(self.environments) > 1:
            self.curr_env =random.choice(self.environments)
            self.layout_name =self.curr_env.base_env.mdp.layout_name

        return self.curr_env.reset()

    def step(self, actions):
        return self.curr_env.step(actions)

    def next_layout(self):
        """
        It switches to the next layout in sequence
        """
        if self.env_idx >= len(self.environments) - 1:
            raise IndexError("No more layouts available")

        self.env_idx += 1
        self.curr_env = self.environments[self.env_idx]
        self.layout_name = self.curr_env.base_env.mdp.layout_name

    def reset_layouts(self):
        """Reset to the first layout in the list"""
        self.env_idx = 0
        self.curr_env = self.environments[self.env_idx]
        self.layout_name = self.curr_env.base_env.mdp.layout_name