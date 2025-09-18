class TrajectoryBuffer:
    """
    We store here all the trajectories collected during the training
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.states =[]
        self.actions =[]
        self.log_probs =[]
        self.rewards =[]
        self.dones =[]

    def store_transition(self,state,action,log_probability,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_probability)
        self.rewards.append(reward)
        self.dones.append(done)