class config():
    
    game_name = "Pong-v0"
    
    history_length = 4
    height=80
    width=80

    #Replay Buffer
    REPLAY_BUFFER_SIZE = 60000 #
    
    
    #main
    skip_frame=4
    MAX_TIME_STEPS = 5000000
    REPLAY_START_SIZE = 50000
    
    target_UPDATE_FREQ = 10000
    LEARNING_FREQ = 4
    
    BATCH_SIZE = 32
    DISCOUNT_FACTOR = 0.99
    
    REWARD_RECORD_FREQ = 25000
    MODEL_RECORD_FREQ = 250000