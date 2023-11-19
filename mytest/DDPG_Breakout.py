# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:01:05 2019

@author: Administrator
"""

import sys
import os
import time
import logging

import cv2  # OPENCV2
import cv2 as cv

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80
CNN_INPUT_DEPTH = 1
SERIES_LENGTH = 4

REWARD_COFF = 3.0

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.0001
MODEL_INITIAL_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.0001

REPLAY_SIZE = 10000

BATCH_SIZE = 32
GAMMA = 0.99
OBSERVE_TIME = 500
ENV_NAME = 'Breakout-v0'
#ENV_NAME = 'Breakout-v4'
EPISODE = 100000
#STEP  = 1500
STEP  = 10000
TEST = 10

SAVE_STEP = 10000
#-----------------
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
#--------------------


class ImageProcess():
    def ColorMat2B(self, state):   # this is the function used for the game flappy bird
        height = 80
        width = 80
        state_gray = cv2.cvtColor( cv2.resize( state, ( height, width ) ) , cv2.COLOR_BGR2GRAY )
        _,state_binary = cv2.threshold( state_gray, 127, 255, cv2.THRESH_BINARY )
        state_binarySmall = cv2.resize( state_binary, ( width, height ))
        cnn_inputImage = state_binarySmall.reshape( ( height, width ) )
        return cnn_inputImage

    def ColorMat2Binary(self, state):
        # state_output = tf.image.rgb_to_grayscale(state_input)
        # state_output = tf.image.crop_to_bounding_box(state_output, 34, 0, 160, 160)
        # state_output = tf.image.resize_images(state_output, 80, 80, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # state_output = tf.squeeze(state_output)
        # return state_output

        height = state.shape[0]
        width = state.shape[1]
        nchannel = state.shape[2]

        sHeight = int(height * 0.5)
        sWidth = CNN_INPUT_WIDTH

        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        # print state_gray.shape
        # cv2.imshow('test2', state_gray)
        # cv2.waitKey(0)

        _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)
        """
        cv2.imshow('test2', state_binary)
        time.sleep(1)
        """
        
        state_binarySmall = cv2.resize(state_binary, (sWidth, sHeight), interpolation=cv2.INTER_AREA)

        cnn_inputImg = state_binarySmall[25:, :]
        # rstArray = state_graySmall.reshape(sWidth * sHeight)
        cnn_inputImg = cnn_inputImg.reshape((CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT))
        # print cnn_inputImg.shape

        return cnn_inputImg

    def ShowImageFromNdarray(self, state, p):
        imgs = np.ndarray(shape=(4, 80, 80))

        for i in range(0, 80):
            for j in range(0, 80):
                for k in range(0, 4):
                    imgs[k][i][j] = state[i][j][k]

        cv2.imshow(str(p + 1), imgs[0])
        cv2.imshow(str(p + 2), imgs[1])
        cv2.imshow(str(p + 3), imgs[2])
        cv2.imshow(str(p + 4), imgs[3])


class DQN():
    def __init__(self, env):
        self.imageProcess = ImageProcess()
        self.epsilon = INITIAL_EPSILON
        self.replay_buffer = deque()
        self.recent_history_queue = deque()
        self.action_dim = env.action_space.n
        self.state_dim = CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH
        self.time_step = 0

        #self.session = tf.InteractiveSession()
        self.session = tf.Session()
        self.create_network()
        # self.create_training_method()
        self.observe_time = 0

        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('/path/to/logs', self.session.graph)

        self.session.run(tf.global_variables_initializer())
        #tf.global_variables_initializer().run()
        
        
    def save_model(self, model_path, model_id, steps=None):
        """
        save model
        """
        if model_path != None:
            saver = tf.train.Saver(max_to_keep=1)
            model_ckpt = os.path.join(model_path, model_id+".ckpt")
            if steps != None:
                saver.save(self.session, model_ckpt, global_step=steps)
                #saver.save(self.session, model_ckpt, global_step=steps, write_meta_graph=False)
            else:
                saver.save(self.session, model_ckpt)
            print("save model to: ", model_ckpt)
            
        
    def restore_model(self, model_path):
        """
        check model
        """
        start_step = 0
        saver = tf.train.Saver()
        if model_path != None:
            ckpt = tf.train.latest_checkpoint(model_path)
            #print("model_ckpt: ", ckpt)
            if ckpt:
                self.epsilon = MODEL_INITIAL_EPSILON
                print("ckpt: ", ckpt)
                saver.restore(self.session, ckpt)
                if ckpt.find('ckpt-')>0:
                    start_step = int(ckpt.split('-')[-1])
                logging.info("restore checkpoint from: {0}".format(ckpt))
                print("restore checkpoint from: {0}".format(ckpt))
                    

    def create_network(self):

        INPUT_DEPTH = SERIES_LENGTH

        self.input_layer = tf.placeholder(tf.float32, [None, CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT, INPUT_DEPTH],
                                          name='status-input')
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])

        W1 = self.get_weights([8, 8, 4, 32])
        b1 = self.get_bias([32])

        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, W1, strides=[1, 4, 4, 1], padding='SAME') + b1)
        conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = self.get_weights([4, 4, 32, 64])
        b2 = self.get_bias([64])

        h_conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)
        # conv2 = tf.nn.max_pool( h_conv2, ksize = [ 1, 2, 2, 1 ], strides= [ 1, 2, 2, 1 ], padding= 'SAME' )

        W3 = self.get_weights([3, 3, 64, 64])
        b3 = self.get_bias([64])

        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)
        # conv3 = tf.nn.max_pool( h_conv3, ksize= [ 1,2,2,1], strides=[ 1,2,2,1 ],padding= 'SAME' )

        W_fc1 = self.get_weights([1600, 512])
        b_fc1 = self.get_bias([512])

        # h_conv2_flat = tf.reshape( h_conv2, [ -1, 11 * 11 * 32 ] )
        conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

        W_fc2 = self.get_weights([512, self.action_dim])
        b_fc2 = self.get_bias([self.action_dim])

        self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))

        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    # def create_training_method(self):
    #
    #   # if len(self.recent_history_queue) > 4:
    #   #   sess = tf.Session()
    #   #   print sess.run(self.Q_value)
    #   # global_step = tf.Variable(0, name='global_step', trainable=True)
    #   # self.optimizer = tf.train.AdamOptimizer( 0.001 ).minimize( self.cost )

    def train_network(self):
        self.time_step += 1

        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]
        # self.imageProcess.ShowImageFromNdarray( state_batch[0], 1 )

        y_batch = []
        #Q_value_batch = self.Q_value.eval(feed_dict={self.input_layer: next_state_batch})
        Q_value_batch = self.session.run(self.Q_value, feed_dict={self.input_layer: next_state_batch})

        # print Q_value_batch
        # print self.time_step
        # cv2.waitKey(0)

        # print Q_value_batch.shape

        for i in range(BATCH_SIZE):

            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.session.run(self.optimizer, feed_dict={

            self.input_layer: state_batch,
            self.action_input: action_batch,
            self.y_input: y_batch
        })


    def percieve(self, state_shadow, action_index, reward, state_shadow_next, done, episode):

        action = np.zeros( self.action_dim )
        action[ action_index ] = 1

        self.replay_buffer.append([state_shadow, action, reward, state_shadow_next, done])

        self.observe_time += 1
        if (self.observe_time % 1000 == 0) and (self.observe_time <= OBSERVE_TIME):
            print(self.observe_time)

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE and self.observe_time > OBSERVE_TIME:
            self.train_network()

    def get_greedy_action(self, state_shadow):
        
        #rst = self.Q_value.eval(feed_dict={self.input_layer: [state_shadow]})[0]
        rst = self.session.run(self.Q_value, feed_dict={self.input_layer: [state_shadow]})[0]
        # print rst
        #print(np.max( rst ))
        return np.argmax(rst)

    def get_action(self, state_shadow):
        if self.epsilon >= FINAL_EPSILON and self.observe_time > OBSERVE_TIME:
            self.epsilon -= (self.epsilon - FINAL_EPSILON) * EPSILON_DECAY_RATE

        action_index = None
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            action_index = self.get_greedy_action(state_shadow)

        return action_index


    def get_weights(self, shape):
        weight = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weight)

    def get_bias(self, shape):
        bias = tf.constant(0.01, shape=shape)
        return tf.Variable(bias)

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


EPISODE = 0
def main():
    basePath = os.getcwd()
    model_base_path = os.path.join(os.path.dirname(basePath), "model")
    if os.path.isdir(model_base_path) == False:
        os.makedirs(model_base_path)
        
    model_id = ENV_NAME+'-DQN'
    model_path = os.path.join(model_base_path, model_id)
    if os.path.isdir(model_path) == False:
        os.makedirs(model_path)   
        
    env = gym.make(ENV_NAME)
    state_shadow = None
    next_state_shadow = None

    agent = DQN(env)

    agent.restore_model(model_path)

    # game_state = game.GameState()
    max_reward = 0
    all_reward = 0
    
    decade_reward = 0
    decade_max_reward = 0
    
    total_step = 0
    for episode in range(EPISODE+1):
        step = 0
        
        total_reward = 0
        state = env.reset()
        state = agent.imageProcess.ColorMat2Binary(state)  # now state is a binary image of 80 * 80
        state_shadow = np.stack((state, state, state, state), axis=2)

        for step in range(STEP):
            total_step += 1
            step += 1
            if episode==EPISODE:
                env.render()
                time.sleep(0.05)
                
            action = agent.get_action(state_shadow)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape( agent.imageProcess.ColorMat2Binary( next_state ), ( 80,80,1 ) )

            # print next_state.shape
            # print state_shadow.shape
            next_state_shadow = np.append( next_state, state_shadow[ :,:,:3 ], axis= 2 )

            total_reward += reward
            agent.percieve(state_shadow, action, reward, next_state_shadow, done, episode)
            state_shadow = next_state_shadow
            """
            if total_step % SAVE_STEP == 0:
                agent.save_model(model_path, model_id)
            """
            if done:
                break
        if total_reward > max_reward:
            max_reward = total_reward
            
        all_reward += total_reward
        
        if total_reward > decade_max_reward:
            decade_max_reward = total_reward
            
        print('Episode:', episode, ', Total Point:', total_reward, ', step:', step, ', epsilon: %.5f'% agent.epsilon)
        decade_reward += total_reward
        if episode>0 and episode % 10 == 0:
            eva_reward = all_reward/episode
            decade_ave_reward = decade_reward/10
            print('-------------')
            print('Decade:', episode/10, 
                  ", Max reward, Decade:{0}/Total:{1}".format(decade_max_reward, max_reward),
                  "  Average Reward, Decade:{0}/Total:{1}".format("%.1f"% decade_ave_reward, "%.1f"% eva_reward))
            print('-------------')
            decade_reward = 0
            decade_max_reward = 0
            """
            save model
            """
            agent.save_model(model_path, model_id)
            
    #agent.save_model(model_path, model_id)

if __name__ == '__main__':
    main()
