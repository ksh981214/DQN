from config import config

import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


#main
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from replay_buffer import ReplayBuffer
from scheduler import e_scheduler
from scheduler import lr_scheduler

class DQNAgent(object):
    def __init__(self, sess, num_actions):
        self.sess = sess
        self.num_actions = num_actions
        self.lr = tf.placeholder(dtype = tf.float32, name = 'lr')
        self.history_length = config.history_length
        self.height = config.height
        self.width = config.width
        
        #self.gamma = gamma
        
        self.build_prediction_network()
        self.build_target_network()
        self.build_training()
        
    def build_prediction_network(self):
        with tf.variable_scope('pred_network'):
            self.state = tf.placeholder(dtype = tf.float32, shape=(None, self.width, self.height, 1 * self.history_length), name = 'state')
            
            model = layers.conv2d(inputs = self.state,
                                    num_outputs = 32,
                                    activation_fn = tf.nn.relu,
                                    stride = 4,
                                    kernel_size = 8,
                                    padding = 'VALID')
            model = layers.conv2d(inputs = model,
                                    num_outputs = 64,
                                    activation_fn = tf.nn.relu,
                                    stride = 2,
                                    kernel_size = 4,
                                    padding = 'VALID')
            model = layers.conv2d(inputs = model,
                                    num_outputs = 64,
                                    activation_fn = tf.nn.relu,
                                    stride = 1,
                                    kernel_size = 3,
                                    padding = 'VALID')
            
            model = layers.flatten(model)
            model = layers.fully_connected(inputs = model,
                                             num_outputs = 512,
                                             activation_fn = tf.nn.relu)                
            self.Q = layers.fully_connected(inputs = model,
                                                 num_outputs = self.num_actions,
                                                 activation_fn = None, scope='Q_values')
            self.Q_action = tf.argmax(self.Q, dimension = 1)
    def build_target_network(self):
        with tf.variable_scope('target_network'):
            self.target_state = tf.placeholder(dtype = tf.float32, shape=(None, self.width, self.height, 1 * self.history_length), name = 'state')
            
            model = layers.conv2d(inputs = self.target_state,
                                    num_outputs = 32,
                                    activation_fn = tf.nn.relu,
                                    stride = 4,
                                    kernel_size = 8,
                                    padding = 'VALID')
            model = layers.conv2d(inputs = model,
                                    num_outputs = 64,
                                    activation_fn = tf.nn.relu,
                                    stride = 2,
                                    kernel_size = 4,
                                    padding = 'VALID')
            model = layers.conv2d(inputs = model,
                                    num_outputs = 64,
                                    activation_fn = tf.nn.relu,
                                    stride = 1,
                                    kernel_size = 3,
                                    padding = 'VALID')
            
            model = layers.flatten(model)
            model = layers.fully_connected(inputs = model,
                                             num_outputs = 512,
                                             activation_fn = tf.nn.relu)
            self.target_Q = layers.fully_connected(inputs = model,
                                             num_outputs = self.num_actions,
                                             activation_fn = None, scope='Q_values')
            
                
    def update_target_network(self):
        #pred_vars = tf.get_collection(
        #    tf.GraphKeys.TRAINABLE_VARIABLES, scope ='pred_network')
        pred_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope ='pred_network')
#           target_vars = tf.get_collection(
#           tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'target_network')
        target_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_network')

        #move the all param from main to tgt
        for pred_var, target_var in zip(pred_vars, target_vars):
            weight_input = tf.placeholder(dtype = tf.float32, name='weight' )
            target_var.assign(weight_input).eval({weight_input: pred_var.eval(session = self.sess)}, session = self.sess)
                
    def build_training(self):
            
        self.target_Q_p = tf.placeholder(dtype = tf.float32, shape=(None), name='target_Q_p')
        self.action = tf.placeholder(dtype = tf.int32, shape=(None), name='action') #batch size개의 행동
        action_one_hot = tf.one_hot(self.action, self.num_actions, 1.0,0.0, name='acion_one_hot')
        q_of_action = tf.reduce_sum(self.Q * action_one_hot, reduction_indices=1, name='q_of_action')
        
        #self.delta = tf.square(self.target_q_t - q_of_action)
        self.loss = tf.reduce_mean(tf.square(self.target_Q_p - q_of_action), name='loss')
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)
        
    def predict_action(self, state):
        action_distribution = self.sess.run(
        self.Q, feed_dict={self.state:[state]})[0]
        
        action = np.argmax(action_distribution)
        
        return action
    
    def process_state_into_stacked_frames(self, frame, past_frames, past_state=None):
        full_state = np.zeros((self.width,self.height,self.history_length)) #84 x 84 x 4
        
        if past_state is not None:
            for i in range(past_state.shape[2]-1):
                #그 전의 과거를 앞으로 한 칸씩 당겨오고 
                full_state[:,:,i] = past_state[:,:,i+1]
            #현재 들어온 프레임을 맨 뒤에 삽입
#             full_state[:,:,-1] = imresize(to_greyscale(frame), (self.width, self.height))/255.0
            #full_state[:,:,-1] = imresize(frame, (self.width, self.height))/255.0
            #print(len(frame[0]))
            full_state[:,:,-1] = np.squeeze(frame)
            
                
        else:
            all_frames = past_frames + [np.squeeze(frame)]
            
            for i, frame_f in enumerate(all_frames):
#                 full_state[:,:,-1] = imresize(to_greyscale(frame), (self.width, self.height))/255.0
                #full_state[:,:,i] = imresize(frame, (self.width, self.height))/255.0
                full_state[:,:,i] = np.squeeze(frame)
            #print(len(frame[0]))
            full_state = full_state.astype('float32')
        
        return full_state
        
        

def main(argv):
    
    env = gym.make(config.game_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),overwrite_render=True)
    num_actions=env.action_space.n
    
    #with tf.Session() as sess:
    sess = tf.Session()
    
    agent = DQNAgent(sess=sess, num_actions=num_actions)
    sess.run(tf.global_variables_initializer())
    
    '''
    episode 10번에 한 번씩 로그를 저장, 그 때 rewards의 평균을 저장
    그런 다음 학습 결과를 저장하기 위해 tf.train.Saver와 텐서플로 세션, 그리고 로그를 저장하기 위한
    tf.summary.FileWriter 객체를 생성
    '''
    rewards = tf.placeholder(dtype = tf.float32, shape=(None), name='reward')#batch size개의 보상
    
    saver = tf.train.Saver()
    tf.summary.scalar('avg.reward/ep', tf.reduce_mean(rewards))
    
    writer = tf.summary.FileWriter('logs_2', sess.graph)
    summary_merged = tf.summary.merge_all()
    
    episode_rewards = [] #에피소드당 리워드 저장
    batch_loss = [] #batch당 loss 저장
    
    replay_buffer = ReplayBuffer()
    time_step = 0
    episode = 0
    total_reward_list = []
    
    #scheduler
    e = e_scheduler()
    lr = lr_scheduler()
    
    while time_step < config.MAX_TIME_STEPS:
        
        done = False #에피소드가 종료되었는가
        total_reward = 0 
        
        frame = env.reset() #한 순간의 화면 (쌓이기 전) 84 x 84 x 1 , np.array

        #past_frames = np.zeros((frame.shape[0],frame.shape[1], agent.history_length - 1))
        past_frames = []
        for i in range(agent.history_length - 1):
            past_frames.append(list(np.squeeze(frame))) # width x weight x 1 이라서 1인 차원을 삭제한 후 list로 변경하고 past_frames에 append

        
        #state --> history length만큼 쌓임
        state = agent.process_state_into_stacked_frames(frame, past_frames, past_state=None)
        
        while not done:
            
            if np.random.rand() < e.get() :
                action = env.action_space.sample()
            else:
                action = agent.predict_action(state)
            
            time_step += 1
            
            frame_after, reward, done, info = env.step(action)
            
            #새로 생긴 frame을 과거 state에 더해줌.
            state_after = agent.process_state_into_stacked_frames(frame_after, past_frames, past_state = state)
            
            past_frames.append(frame_after) #이제 history length만큼 됨.
            past_frames = past_frames[-config.history_length:]
            
            #replay_buffer.add_experience(state, action, reward, state_after, done)
            replay_buffer.add_experience(state, action, reward, done)
            
            state = state_after
            
            total_reward += reward
            
            if time_step > config.REPLAY_START_SIZE and time_step % config.LEARNING_FREQ == 0:
                e.update(time_step)
                lr.update(time_step)
                #dqn.train(lr.get())
                b_state, b_action, b_reward, b_state_after, b_done = replay_buffer.sample_batch(config.BATCH_SIZE)
                
                Q_of_state_after = agent.target_Q.eval(feed_dict={agent.target_state: b_state_after}, session = agent.sess)
                
                #max_Q_of_state_after = np.max(Q_of_state_after, axis=1)
                
                target_Q_p = []
                for i in range(config.BATCH_SIZE):
                    if b_done[i]:
                        target_Q_p.append(b_reward[i])
                    else:
                        target_Q_p.append(b_reward[i] + config.DISCOUNT_FACTOR * np.max(Q_of_state_after[i]))
                        
                agent.sess.run([agent.train_step, agent.Q, agent.loss], {
                    agent.target_Q_p : target_Q_p,
                    agent.action: b_action,
                    agent.state: b_state,
                    agent.lr: lr.get()
                })
                
            if time_step % config.target_UPDATE_FREQ == 0:
                agent.update_target_network()
                
            if time_step % config.REWARD_RECORD_FREQ == 0 and len(total_reward_list) != 0:
                #print("로그를 저장합니다.")
                summary = sess.run(summary_merged,
                                  feed_dict = {rewards: total_reward_list})
                writer.add_summary(summary, time_step)
                total_reward_list =[]
        
            if time_step % config.MODEL_RECORD_FREQ == 0:
                saver.save(sess, 'model/dqn.ckpt', global_step = time_step)
        
        
        #학습과 상관 x
        episode += 1
        #For Debugging
        if episode % 100 == 0:
            print('episode : %d 점수: %d' % (episode, total_reward))
        
        total_reward_list.append(total_reward)
        #print(time_step)
#         if time_step % config.REWARD_RECORD_FREQ == 0:
#             #print("로그를 저장합니다.")
#             summary = sess.run(summary_merged,
#                               feed_dict = {rewards: total_reward_list})
#             writer.add_summary(summary, time_step)
#             total_reward_list =[]
        
#         if time_step % config.MODEL_RECORD_FREQ == 0:
#             saver.save(sess, 'model/dqn.ckpt', global_step = time_step)
            
            
# def main(_):
#     agent_play_and_train()
#     tf.app.run()
if __name__ == '__main__':
    tf.app.run()