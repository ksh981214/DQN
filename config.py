class config():
    
    #game_name = "Pong-v0"
    game_name="PongNoFrameskip-v4"
    preprocess = True #'Preprocess atari environments to 84*84 frames'
    scale = False # 'Scale observations into [0,1]'
    episode_life = True # 'Make end-of-life == end-of-episode'
    clip_rewards = True # 'Bin rewards to [-1, 0, +1] by its sign'
    no_op_reset = True # 'Sample initial states by taking random number of no-ops on reset'
    max_and_skip = True # 'Skip frames to accelerate the learning process'
    observation_dims = [84,84,1] # 'Observation dimensions of the environment'
    
    history_length = 4
    height=84
    width=84

    #Replay Buffer
    REPLAY_BUFFER_SIZE = 100000 #
    
    
    #main
    skip_frame=4
    MAX_TIME_STEPS = 2500000
    REPLAY_START_SIZE = 10000
    
    target_UPDATE_FREQ = 10000
    LEARNING_FREQ = 4
    
    BATCH_SIZE = 32
    DISCOUNT_FACTOR = 0.99
    
    REWARD_RECORD_FREQ = 10000
    MODEL_RECORD_FREQ = 100000
    
    