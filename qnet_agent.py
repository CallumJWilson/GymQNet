import tensorflow as tf
import numpy as np
import random
import math
from environment import Environment
from operator import itemgetter
import pdb
import time

class QNetAgent():
	def __init__(self,agent_config,network_config,env):

		self.alpha=network_config.alpha
		self.clip_norm=network_config.clip_norm
		self.update_steps=network_config.update_steps
		self.timing=network_config.timing

		if isinstance(network_config.N_hid,int):
			self.N_hid = network_config.N_hid
		else:
			self.N_hid = self.action_size

		self.gamma=agent_config.gamma
		self.eps=agent_config.eps0
		self.eps0=agent_config.eps0
		self.epsf=agent_config.epsf
		self.n_eps=agent_config.n_eps
		self.minib=agent_config.minib
		self.max_mem=agent_config.max_mem

		self.state_size = env.state_size
		self.action_size = env.action_size

		self.state_input = tf.placeholder(shape=[1,self.state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random_uniform([self.state_size,self.N_hid],0,0.1))
		self.b_in = tf.Variable(tf.random_uniform([1,self.N_hid],0,0))
		self.W = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,0.1))

		act_fn = tf.tanh
		self.act = act_fn(tf.matmul(self.state_input,tf.add(self.w_in,self.b_in)), name=None)
		self.Q_est = tf.matmul(self.act,self.W)

		self.nextQ = tf.placeholder(shape=[1,self.action_size],dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(self.nextQ - self.Q_est))
		trainer = tf.train.RMSPropOptimizer(self.alpha)
		if isinstance(self.clip_norm, float):
			grads = trainer.compute_gradients(loss,[self.W,self.w_in,self.b_in])
			cap_grads = [(tf.clip_by_norm(grad, self.clip_norm), var) for grad, var in grads]
			self.updateModel = trainer.apply_gradients(cap_grads)
		else:
			self.updateModel = trainer.minimize(loss,var_list=[self.W,self.w_in,self.b_in])

		# target network
		self.wt = tf.Variable(tf.random_uniform([self.state_size,self.N_hid],0,0.1))
		self.bt = tf.Variable(tf.random_uniform([1,self.N_hid],0,0.01))
		self.Wt = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,0.1))
		self.actt = act_fn(tf.matmul(self.state_input,tf.add(self.wt,self.bt)), name=None)
		self.Qt = tf.matmul(self.actt,self.Wt)
		
		self.wt_assign = self.wt.assign(self.w_in)
		self.bt_assign = self.bt.assign(self.b_in)
		self.Wt_assign = self.Wt.assign(self.W)

		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())

		# Memory initialisation
		self.prev_s=[]
		self.prev_a=[]
		self.memory=[]
		self.step_count=0
		self.ep_no=0

	def action_select(self,env,state):

		if np.random.random(1)<self.eps:
			action=env.rand_action()
		else:
			q_s=self.sess.run(self.Q_est,feed_dict={self.state_input:state.reshape(1,-1)})
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		return action

	def update_net(self,state,reward,done):

		# Update epsilon
		if done:
			state=[]
			self.ep_no+=1
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf

		# Update epsilon
		self.memory.append([self.prev_s,self.prev_a,reward,state])
		if len(self.memory)>self.max_mem:
			del self.memory[0]

		# Select data from memory
		if len(self.memory)>self.minib & self.minib>1:
			sample_ind=random.sample(range(1,len(self.memory)),self.minib)
		elif self.minib==1:
			sample_ind=[len(self.memory)-1]
		else:
			sample_ind=range(len(self.memory))

		# Update network using selected data
		with self.sess.as_default():
			t0=time.time()
			for ind in sample_ind:
				s=self.memory[ind][0]
				r=self.memory[ind][2]
				a=self.memory[ind][1]
				q_s=self.Qt.eval({self.state_input:s.reshape(1,-1)})
				if len(self.memory[ind][3])>0:
					q_dash=self.Qt.eval({self.state_input:self.memory[ind][3].reshape(1,-1)})
					q_s[0][a]=r+self.gamma*np.amax(q_dash)
				else:
					q_s[0][a]=r
				s_update=s.reshape(1,-1)
				q_target=q_s.reshape(1,-1)
				self.sess.run(self.updateModel,{self.state_input:s_update,self.nextQ:q_target})
			t1=time.time()

		# Update target network
		self.step_count += 1
		if self.step_count>=self.update_steps:
			self.sess.run(self.wt_assign)
			self.sess.run(self.bt_assign)
			self.sess.run(self.Wt_assign)
			self.step_count=0
		if self.timing:
			return t1-t0
