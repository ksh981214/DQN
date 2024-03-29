from config import config

import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

#main
from utils.wrappers import wrap_deepmind
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

        self.build_prediction_network()
        self.build_target_network()
        self.build_training()
        
        #for update network
        pred_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope ='pred_network')
        target_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_network')
    
        self.update_fn = tf.group(*[tf.assign(target_vars[i], pred_vars[i]) for i in range(len(target_vars))])
        
    def build_prediction_network(self):
        with tf.variable_scope('pred_network'):
            self.state = tf.placeholder(dtype = tf.float32, shape=(None, self.width, self.height, 1 * self.history_length), name = 'state')
            
            self.conv1 = layers.conv2d(inputs = self.state,
                                    num_outputs = 32,
                                    activation_fn = tf.nn.relu,
                                    stride = 4,
                                    kernel_size = 8,
                                    padding = 'VALID')
            self.conv2 = layers.conv2d(inputs = self.conv1,
                                    num_outputs = 64,
                                    activation_fn = tf.nn.relu,
                                    stride = 2,
                                    kernel_size = 4,
                                    padding = 'VALID')
            self.conv3 = layers.conv2d(inputs = self.conv2,
                                    num_outputs = 64,
                                    activation_fn = tf.nn.relu,
                                    stride = 1,
                                    kernel_size = 3,
                                    padding = 'VALID')
            
            self.fc1 = layers.flatten(self.conv3)
            self.fc2 = layers.fully_connected(inputs = self.fc1,
                                             num_outputs = 512,
                                             activation_fn = tf.nn.relu)                
            self.Q = layers.fully_connected(inputs = self.fc2,
                                                 num_outputs = self.num_actions,
                                                 activation_fn = None)

    def build_target_network(self):
        with tf.variable_scope('target_network'):
            self.target_state = tf.placeholder(dtype = tf.float32, shape=(None, self.width, self.height, 1 * self.history_length), name = 'target_state')
            
            self.t_conv1 = layers.conv2d(inputs = self.target_state,
                                    num_outputs = 32,
                                    activation_fn = tf.nn.relu,
                                    stride = 4,
                                    kernel_size = 8,
                                    padding = 'VALID')
            self.t_conv2 = layers.conv2d(inputs = self.t_conv1,
                                    num_outputs = 64,
                                    activation_fn = tf.nn.relu,
                                    stride = 2,
                                    kernel_size = 4,
                                    padding = 'VALID')
            self.t_conv3 = layers.conv2d(inputs = self.t_conv2,
                                    num_outputs = 64,
                                    activation_fn = tf.nn.relu,
                                    stride = 1,
                                    kernel_size = 3,
                                    padding = 'VALID')
            
            self.t_fc1 = layers.flatten(self.t_conv3)
            self.t_fc2 = layers.fully_connected(inputs = self.t_fc1,
                                             num_outputs = 512,
                                             activation_fn = tf.nn.relu)
            self.target_Q = layers.fully_connected(inputs = self.t_fc2,
                                             num_outputs = self.num_actions,
                                             activation_fn = None)
            
            
    def build_training(self):
            
        self.target_Q_p = tf.placeholder(dtype = tf.float32, shape=(None), name='target_Q_p')
        self.action = tf.placeholder(dtype = tf.int32, shape=(None), name='action') #batch size개의 행동
        action_one_hot = tf.one_hot(self.action, self.num_actions, 1.0,0.0, name='acion_one_hot')
        q_of_action = tf.reduce_sum(self.Q * action_one_hot, reduction_indices=1, name='q_of_action')
        
        
        self.delta = self.target_Q_p - q_of_action
        self.loss = tf.reduce_mean(tf.where(tf.abs(self.delta)<1.0,tf.square(self.delta)*0.5, tf.abs(self.delta)-0.5), name='loss')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.optimizer,10.0)
        
        self.train_step = self.optimizer.minimize(self.loss)

    def predict_action(self, state):
        action_distribution = self.sess.run(
        self.Q, feed_dict={self.state:[state]})[0]
        action = np.argmax(action_distribution)
        
        return action
    
    def process_state_into_stacked_frames(self, frame, past_frames, past_state=None):
        if past_state is not None:
            full_state = np.concatenate((past_state, frame), axis=2)[:,:,1:]                
        else:  #None
            full_state = np.concatenate((past_frames,frame), axis=2)             
        return full_state
    
def main(argv):
    
    env = gym.make(config.game_name)
    env = wrap_deepmind(env, config.episode_life, config.preprocess, config.max_and_skip,config.clip_rewards, config.no_op_reset, config.scale)
    
    num_actions=env.action_space.n
    
    sess = tf.Session()
    
    agent = DQNAgent(sess=sess, num_actions=num_actions)
    
    sess.run(tf.global_variables_initializer())

    rewards = tf.placeholder(dtype = tf.float32, shape=(None), name='reward')
    
    saver = tf.train.Saver()
    tf.summary.scalar('avg.reward/ep', tf.reduce_mean(rewards))
    tf.summary.scalar('max.reward/ep', tf.reduce_max(rewards))
    
    writer = tf.summary.FileWriter('logs_12_v4_allwrap_constant_lr', sess.graph)
    summary_merged = tf.summary.merge_all()
    
    episode_rewards = []
    batch_loss = [] 
    
    replay_buffer = ReplayBuffer()
    time_step = 0
    episode = 0
    total_reward_list = []
    
    #scheduler
    e = e_scheduler()
    lr = lr_scheduler()
    
    while time_step < config.MAX_TIME_STEPS:
        
        done = False 
        total_reward = 0 
        
        '''
        frame --> 84 x 84 x 1
        state --> 84 x 84 x 4
        '''
        
        
        frame = env.reset() 

        frame_scale = np.array(frame).astype(np.float32) / 255.0

        #맨 처음 frame을 받아올때는 past_frames이 존재하지않으므로, (84x84)의 0인 행렬을 받아서 초기화
        past_frames = np.zeros((config.height,config.width,agent.history_length - 1), dtype=np.uint8) #저장용
        past_frames_scale = np.zeros((config.height,config.width,agent.history_length - 1), dtype=np.float32) #학습용

        state = agent.process_state_into_stacked_frames(frame, past_frames, past_state=None)
        state_scale = np.array(state).astype(np.float32) / 255.0
        
        while not done:
            
            if np.random.rand() < e.get() or time_step < config.REPLAY_START_SIZE:
                action = env.action_space.sample()
            else:
                action = agent.predict_action(state_scale)
            time_step += 1
            
            frame_after, reward, done, info = env.step(action)

            frame_after_scale = np.array(frame_after).astype(np.float32) / 255.0

            replay_buffer.add_experience(state, action, reward, done)

            if not done: #+21 or -21

                #새로 생긴 frame을 과거 state에 더해줌.
                state_after = agent.process_state_into_stacked_frames(frame_after, past_frames, past_state = state)

                state_after_scale = np.array(state_after).astype(np.float32) / 255.0
                
                past_frames = np.concatenate((past_frames, frame_after), axis=2)
                past_frames = past_frames[:,:,1:]
                
                past_frames_scale = np.array(past_frames).astype(np.float32) / 255.0
                
                #print(past_frames.shape)
                state = state_after
                state_scale = state_after_scale

            total_reward += reward
            
            #training
            if time_step > config.REPLAY_START_SIZE and time_step % config.LEARNING_FREQ == 0:
                e.update(time_step)
                lr.update(time_step)

                b_state, b_action, b_reward, b_state_after, b_done = replay_buffer.sample_batch(config.BATCH_SIZE)
                
                Q_of_state_after = agent.sess.run(agent.target_Q,
                                                 feed_dict={agent.target_state: b_state_after})

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
                agent.sess.run(agent.update_fn)
                
            if time_step % config.REWARD_RECORD_FREQ == 0 and len(total_reward_list) != 0:
                summary = sess.run(summary_merged,
                                  feed_dict = {rewards: total_reward_list})
                writer.add_summary(summary, time_step)
                total_reward_list =[]
        
            if time_step % config.MODEL_RECORD_FREQ == 0:
                saver.save(sess, 'model_12_v4_allwrap_constant_lr/dqn.ckpt', global_step = time_step)
        
        
        #학습과 상관 x
        episode += 1
        #For Debugging
        if episode % 100 == 0:
            print('episode : %d 점수: %d' % (episode, total_reward))
        
        total_reward_list.append(total_reward)

            
if __name__ == '__main__':
    tf.app.run()