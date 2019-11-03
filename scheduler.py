from config import config
class e_scheduler():
    
    Initial_e = 1
    Final_e = 0.1
    FINAL_EXPLORATION_STEPS = 500000 #2015 Atari_DQN, if EP_STEP == 1000000,. e = 0.1
    
    
    def __init__(self):
        self.e = self.Initial_e
        
    def update(self, time_step):
        if time_step > self.FINAL_EXPLORATION_STEPS :
            self.e = self.Final_e
        else:
            self.e = self.Initial_e - (((self.Initial_e - self.Final_e) / self.FINAL_EXPLORATION_STEPS) * time_step)
        
    def get(self):
        return self.e
    
#NOT USING
class lr_scheduler():
    
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = config.MAX_TIME_STEPS/2
    
   
    
    def __init__(self):
        self.lr = self.lr_begin
    def update(self, time_step):
        if time_step > self.lr_nsteps :
            self.lr = self.lr_end
        else:
            self.lr = self.lr_begin - (((self.lr_begin - self.lr_end) / self.lr_nsteps) * time_step)
    def get(self):
        return self.lr