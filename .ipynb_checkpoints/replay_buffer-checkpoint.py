from config import config
import numpy as np
import random

class ReplayBuffer(object):
    
    def __init__(self):
        self.buffer_size = config.REPLAY_BUFFER_SIZE
        self.state_shape = [config.width, config.height, config.history_length]
        
        self.next_idx = 0
        self.num_in_buffer = 0
        self.states = np.empty([self.buffer_size] + self.state_shape, dtype=np.uint8)
        self.actions = np.empty([self.buffer_size], dtype=np.int32)
        self.rewards = np.empty([self.buffer_size], dtype=np.float32)
        self.dones = np.empty([self.buffer_size], dtype=np.bool)
        
        #print(self.states.shape)
        #print(self.actions.shape)
        #print(self.rewards.shape)
        #print(self.dones.shape)
        
        
    def add_experience(self, state, action, reward, done):
        
        self.states[self.next_idx] = state
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.dones[self.next_idx] = done
        
        self.next_idx = (self.next_idx + 1) % self.buffer_size
        self.num_in_buffer = min(self.buffer_size, self.num_in_buffer + 1) 
        
        #if self.num_in_buffer % 1000 == 0:
        #    print(self.num_in_buffer)

    def sample_batch(self, batch_size):
        
        if self.num_in_buffer < batch_size:
            assert('Replay buffer does not has enough data to sample')
        
        rands = np.arange(self.num_in_buffer-1) #state_after때문에
        
        np.random.shuffle(rands)
        rands = rands[:batch_size]
        
        for i, num in enumerate(rands):
            if self.dones[num]:
                rands[i] -= 1
        
        rands_add_1 = np.add(rands,1)    
        
        #print("스케일 전:", self.states[rands])
        state = np.array(self.states[rands]/255.0, dtype=np.float32)
        #print("스케일 후: ",state)
        action = self.actions[rands]
        reward = self.rewards[rands]
        state_after = np.array(self.states[rands_add_1]/255.0, dtype=np.float32)
        done = np.array([1.0 if self.dones[num] else 0.0 for num in rands],dtype=np.float32)
        
        return state,action,reward,state_after,done