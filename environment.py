# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
import gym


class Environment():
	def __init__(self,env_name='CartPole-v0'):
		self.gym_env = gym.make(env_name)
		self.state_size = np.shape(self.gym_env.observation_space)[0]
		self.action_size = self.gym_env.action_space.n
	def render(self):
		self.gym_env.render()
	def rand_action(self):
		return self.gym_env.action_space.sample()
	def reset(self):
		return self.gym_env.reset()
	def step(self,a,render=False):
		s,r,d,info = self.gym_env.step(a)
		if isinstance(s,int):
			state = np.zeros(self.state_size)
			state[s] = 1
			s = state
		if render:
			self.gym_env.render()
		return s,r,d,info
