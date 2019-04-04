import numpy as np
from qnet_agent import DoubleQNet
from environment import Environment

# import matplotlib.pyplot as plt
# from matplotlib import animation
import time
import pdb


def data_smooth(data,n_avg):
	ind_vec = np.arange(n_avg,len(data)+1,n_avg)
	data_avg = [0]
	for ind in ind_vec:
		data_avg.append(np.mean(data[ind-n_avg:ind]))
	return data_avg


def display_frames_as_gif(frames, filename_gif = None):
	"""
	Displays a list of frames as a gif, with controls
	"""
	plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
	patch = plt.imshow(frames[0])
	plt.axis('off')
	
	def animate(i):
		patch.set_data(frames[i])
	
	anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
	if filename_gif: 
		anim.save(filename_gif, writer='imagemagick', fps=30)


def show_policy(N_ep, env, agent, fname = None):
	frames = []
	observation = env.reset()
	firstframe = env.gym_env.render(mode = 'rgb_array')
	fig,ax = plt.subplots()
	im = ax.imshow(firstframe)
	for ep_no in range(5):
		observation = env.reset()
		done = False
		while not done:
			action = agent.action_select(env,observation)
			observation, _, done, _ = env.step(action)
			frame = env.gym_env.render(mode = 'rgb_array')
			im.set_data(frame)
			frames.append(frame)
	if fname:
		display_frames_as_gif(frames, filename_gif=fname)

def save_data(R_ep, agent, fname):

	try:
		old_data = np.load(fname).tolist()
	except:
		old_data = {'R_run':[], 'agents':[]}
		old_data['netcon'] = vars(NetworkConfig())
		old_data['agentcon'] = vars(AgentConfig())
		# agent type?
	
	agent_data = {}
	agent_data['W'] = agent.sess.run(agent.W)
	agent_data['w_in'] = agent.sess.run(agent.w_in)
	agent_data['b_in'] = agent.sess.run(agent.b_in)

	old_data['R_run'].append(R_ep)
	old_data['agents'].append(agent_data)

	np.save(fname, old_data)


def plot_rewards(R_run, N_avg = 100):
	for R_ep in R_run:
		R_plot=data_smooth(R_ep,N_avg)
		plt.plot(np.arange(len(R_plot))*N_avg,R_plot)
	plt.xlabel('Episode', fontsize=12)
	plt.ylabel('Average Total Discounted Reward', fontsize=12)
	plt.show()


def do_run(N_ep = 1000, run_no = 0, env_name = 'CartPole-v0'):
	env = Environment(env_name)
	agent = EQLMAgent(AgentConfig(), NetworkConfig(), env)

	R_ep = []
	for ep_no in range(N_ep):
		print('Run: ' + repr(try_no) + ' Episode: ' + repr(ep_no))
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


class NetworkConfig():
	def __init__(self):
		self.alpha = 0.01
		self.clip_norm = 1.0
		self.update_steps = 40
		self.gamma_reg = 0.05
		self.N_hid = 14
		self.timing = False


class AgentConfig():
	def __init__(self):
		self.gamma = 0.9
		self.eps0 = 0.95
		self.epsf = 0.01
		self.n_eps = 400
		self.minib = 18
		self.max_mem = 10000
		self.prioritized = False
		self.printQ = False


N_ep = 1000
N_run = 2
env_name = 'CartPole-v0'

R_run = []
agent_run = []
for try_no in range(N_run):
	R_ep, agent, env = do_run(N_ep, try_no, env_name)
	agent_run.append(agent)
	R_run.append(R_ep)
