from config import config
import numpy as np
import random

class ReplayBuffer(object):
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        #self.states_after = []
        self.dones = []
        #self.discounted_returns = []
        
        self.buffer_size = config.REPLAY_BUFFER_SIZE
        
    #def add_experience(self, state, action, reward, state_after, discounted_return):
    #def add_experience(self, state, action, reward, state_after, done):
    def add_experience(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        #self.states_after.append(state_after)
        self.dones.append(done) 
        #self.discounted_returns += discounted_return
        if len(self.states) > self.buffer_size:
            self.delete_old_experiences()
        
    
    def delete_old_experiences(self):
        self.states = self.states[-self.buffer_size:]
        self.actions = self.actions[-self.buffer_size:]
        self.rewards = self.rewards[-self.buffer_size:]
        #self.states_after = self.states_after[-self.buffer_size:]
        self.dones = self.dones[-self.buffer_size:]

    def sample_batch(self, batch_size):
        state, action, reward, state_after, done = [],[],[],[],[]
        #state, action, reward, done = [],[],[],[]
        
        rands = np.arange(len(self.states)-1) #state_after때문에
        np.random.shuffle(rands)
        rands = rands[:batch_size]
        
        for i in rands:
            state.append(self.states[i])
            action.append(self.actions[i])
            reward.append(self.rewards[i])
            state_after.append(self.states[i+1])
            done.append(self.dones[i])
            #terminal.append(self.discounted_returns[i])
        #print((np.array(state_after)).shape)
        return np.array(state),np.array(action),np.array(reward),np.array(state_after),np.array(done)
        #return np.array(state),np.array(action),np.array(reward),np.array(state_after),np.array(done)
            