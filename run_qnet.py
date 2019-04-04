# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
from qnet_agent import QNetAgent
from environment import Environment
import matplotlib.pyplot as plt
import pdb


def data_smooth(data,n_avg):
	ind_vec = np.arange(n_avg,len(data)+1,n_avg)
	data_avg = [0]
	for ind in ind_vec:
		data_avg.append(np.mean(data[ind-n_avg:ind]))
	return data_avg


def show_policy(N_ep, env, agent):
	for ep_no in range(N_ep):
		observation = env.reset()
		env.render()
		done = False
		while not done:
			action = agent.action_select(env,observation)
			observation, _, done, _ = env.step(action)
			env.render()


def plot_rewards(R_run, N_avg = 100):
	for R_ep in R_run:
		R_plot=data_smooth(R_ep,N_avg)
		plt.plot(np.arange(len(R_plot))*N_avg,R_plot)
	plt.xlabel('Episode', fontsize=12)
	plt.ylabel('Average Total Discounted Reward', fontsize=12)
	plt.show()


def do_run(N_ep = 1000, run_no = 0, env_name = 'CartPole-v0'):
	env = Environment(env_name)
	agent = QNetAgent(agent_config(), network_config(), env)

	R_ep = []
	for ep_no in range(N_ep):
		print('Run: ' + repr(run_no) + ' Episode: ' + repr(ep_no))
		observation = env.reset()
		done = False
		r = 0
		while not done:
			action = agent.action_select(env,  observation)
			observation, reward, done, info = env.step(action)
			agent.update_net(observation, reward, done)
			r += reward
		R_ep.append(r)
		print('R: ' + repr(r))
	return R_ep, agent, env


def network_config():
	netcon = {}
	netcon['alpha'] = 0.01
	netcon['clip_norm'] = 1.0
	netcon['update_steps'] = 40
	netcon['N_hid'] = 14
	return netcon


def agent_config():
	agentcon = {}
	agentcon['gamma'] = 0.9
	agentcon['eps0'] = 0.95
	agentcon['epsf'] = 0.01
	agentcon['n_eps'] = 400
	agentcon['minib'] = 20
	agentcon['max_mem'] = 10000
	return agentcon

def main():
	N_ep = 1000
	N_run = 2
	env_name = 'CartPole-v0'

	R_run = []
	agent_run = []
	for run_no in range(N_run):
		R_ep, agent, env = do_run(N_ep, run_no, env_name)
		agent_run.append(agent)
		R_run.append(R_ep)

if __name__ == '__main__':
	main()
