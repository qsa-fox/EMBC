import torch
import numpy as np
import gym
import d4rl
from dataset_utils import get_datasets_state_bc
from gcsl import learn_gcsl_actor, behavior_cloning, learn_reachability
from planning_graph import RoutePlannerBC, RoutePlannerDijkstra
import json
import os
import sys
from gcsl import DiagGaussianPolicy
import matplotlib.pyplot as plt
from construct_graph import construct_graph_reachability_batch
plt.ion()

class PlannerActor:
	def __init__(self, gcsl_actor, graph_planner, config, states=None) -> None:
		self.actor = gcsl_actor
		self.graph_planner = graph_planner
		self.config = config
		self.previous_subgoal = None
		self.subgoal_steps = 0
		self.states = states
		# self.distance_log = []
	def act(self, state, goal, gcsl=False):
		if gcsl:
			pass
		else:
			# cur_pos = state[self.config['valid_state_indices']]
			cur_pos = state
			sub_goal, route_idx = self.graph_planner.plan(cur_pos, goal) 
			# self.distance_log.append(distance)
			# print(f'route_idx: {route_idx}')
			debug = False
			if debug:
				route = self.graph_planner.centers[route_idx]
				plt.scatter(self.states[:, 0], self.states[:, 1])
				plt.xlim(-5, 35), plt.ylim(-5, 35)
				plt.scatter(route[:, 0], route[:, 1])
				plt.scatter(cur_pos[0], cur_pos[1])
				for i in range(len(route)-1):
					p1 = [route[i][0], route[i+1][0]]
					p2 = [route[i][1], route[i+1][1]]
					plt.plot(p1, p2, 'g', linewidth=0.3, alpha=0.5)
				plt.show()
				plt.pause(0.001),plt.clf()
    
				# task_finished = [task_complete(obs) for obs in route]
			if config.get('gcsl_net_delta_target', False):
				delta_sub_goal = sub_goal - cur_pos
				state = np.concatenate([cur_pos, delta_sub_goal], axis=-1)
			else:
				state = np.concatenate([cur_pos, sub_goal], axis=-1)
			act = self.actor.act(state, device='cuda')
			return act

class distance_reachable_func:
	def __init__(self):
		super().__init__()
	
	@staticmethod
	def predict(gc_state):
		return torch.square(gc_state[:, :gc_state.shape[-1]//2]-gc_state[:, gc_state.shape[-1]//2:]).sum(-1)

class xy_distance_reachable_func:
	def __init__(self):
		super().__init__()
	
	@staticmethod
	def predict(gc_state):
		return torch.square(gc_state[:, :gc_state.shape[-1]//2][:, :2]-gc_state[:, gc_state.shape[-1]//2:][:, :2]).sum(-1)

def get_composite_demonstrations(states, goal, config, n_demos=10):
	env = config['env']
	env_name = config['env_name']
	reachable_func = torch.load(f"models/{env_name}/reachability_net-horizon{config['reachability_net_horizon']}-seed{config['seed']}.pth")
	# reachable_func = distance_reachable_func
	connect_chains = construct_graph_reachability_batch(states, reachable_func=reachable_func.predict, max_horizon=2)
	graph_planner = RoutePlannerDijkstra(states, connect_chains, goal, reachable_func=reachable_func)
	demos = []
	for i in range(n_demos):
		obs = env.reset()
		route = graph_planner.get_shortest_route(obs)
		demos.append(route)
	return demos

def build_planner_components(env_name, config):
	if not config['load_model']:
		# ## preprocess datasets
		sequence, states, goal, goal_idx, states_end_idx = get_datasets_state_bc(env_name, config, evaluation_flag=False)
		sequence_mean_score = np.mean([seq['rewards'].sum() for seq in sequence])
		sequence_normalized_score = d4rl.get_normalized_score(env_name, sequence_mean_score)
		print(f'sequence_normalized_score: {sequence_normalized_score}')
	
	## behavior cloning
	# sequence, states, goal, goal_idx, states_end_idx = get_datasets_state_bc(env_name, config, evaluation_flag=False)
	# print('learning bc actor...')
	# config['env_name'] = env_name
	# bc_actor = behavior_cloning(sequence, config=config)
	# torch.save(bc_actor, f"models/{env_name}/bc_actor-seed{config['seed']}.pth")
	# bc_actor = torch.load(f"models/{env_name}/bc-full-seed{config['seed']}.pth")
	# score = evaluate_bc(config['env'], bc_actor, n_episode=100)
	# return score
	
	if not config['load_model']:
		## learning goal-conditioned supervised learning actor
		print('learning goal-conditioned supervised learning actor...')
		gcsl_actor = learn_gcsl_actor(sequence, config=config)
		torch.save(gcsl_actor, f"models/{env_name}/gc_actor-horizon{config['gcsl_net_horizon']}-seed{config['seed']}.pth")
	
	if not config['load_model']:
		# learning reachability_net
		print('learning reachability_net...')
		reachability = learn_reachability(sequence, config=config)
		torch.save(reachability, f"models/{env_name}/reachability_net-horizon{config['reachability_net_horizon']}-seed{config['seed']}.pth")

	sequence, states, goal, _, states_end_idx = get_datasets_state_bc(env_name, config, evaluation_flag=True)
	sequence_mean_score = np.mean([seq['rewards'].sum() for seq in sequence])
	sequence_normalized_score = d4rl.get_normalized_score(env_name, sequence_mean_score)
	print(f'sequence_normalized_score: {sequence_normalized_score}')
 
	# '''synthesizing expert trajectories'''
	# composite_demos = get_composite_demonstrations(states, goal, config, n_demos=1)
	# best_seq_len = [len(seq) for seq in composite_demos]
	# states_end_idx = np.cumsum(best_seq_len, axis=-1)
	# states = np.concatenate(composite_demos)
	# sequence = composite_demos
	# plt.scatter(composite_demos[0][:, 0], composite_demos[0][:, 1]), plt.show()
 
 
	gcsl_actor = torch.load(f"models/{env_name}/gc_actor-horizon{config['gcsl_net_horizon']}-seed{config['seed']}.pth")
	gcsl_actor.prev_evaluate_state = None
	reachable_func = torch.load(f"models/{env_name}/reachability_net-horizon{config['reachability_net_horizon']}-seed{config['seed']}.pth")
	# reachable_func = distance_reachable_func
	graph_planner = RoutePlannerBC(states, states_end_idx, reachable_func=reachable_func)
	actor = PlannerActor(gcsl_actor, graph_planner, config, states=states)
	score = evaluate(env_name, config['env'], actor, goal, n_episode=100, render=False)
	return score
	
def evaluate(env_name, env, actor, goal, n_episode=50, render=False):
	reward_log, score_log, episode_log, distance_log = [], [], [], []
	# env = gym.make(env_name)
	for i in range(n_episode):
		state = env.reset()
		actor.previous_subgoal = None
		actor.subgoal_steps = 0
		done = False
		step, epsiode_r = 0, 0
		while not done:
			action = actor.act(state, goal, gcsl=False)
			state, reward, done, _ = env.step(action)
			step += 1
			epsiode_r += reward
			if render:
				env.render()
		reward_log.append(epsiode_r)
		actor.distance_log = []
		if 'cloned' in env_name or 'human' in env_name:
			if 'v1' not in env_name:
				env_name += '-v1'
		normalized_score = d4rl.get_normalized_score(env_name, epsiode_r)
		score_log.append(normalized_score)
		print(f'env: {env_name}, iter: {i}, episode_r: {normalized_score}, mean_reward: {np.mean(score_log)}')
	episode_log.append(np.mean(score_log))
	print(f'env_name: {env_name}, scores: {episode_log}, {np.sum(episode_log) * 100:.1f}')
	return np.sum(episode_log) * 100
	

def evaluate_bc(env, actor, n_episode=50, render=False):
	reward_log, score_log, episode_log = [], [], []
	for i in range(n_episode):
		# env = gym.make(env_name)
		state = env.reset()
		done = False
		step, epsiode_r = 0, 0
		while not done:
			action = actor.act(state, device='cuda')
			state, reward, done, _ = env.step(action)
			step += 1
			epsiode_r += reward
			if render:
				env.render()
		reward_log.append(epsiode_r)

		normalized_score = d4rl.get_normalized_score(env_name, epsiode_r)
		score_log.append(normalized_score)
		print(f'env: {env_name}, iter: {i}, episode_r: {normalized_score}, mean_reward: {np.mean(score_log)}')
	episode_log.append(np.mean(score_log))
	print(f'env_name: {env_name}, scores: {episode_log}, {np.sum(episode_log) * 100:.1f}')
	return np.sum(episode_log) * 100

def set_seed(env, seed=0):
	torch.manual_seed(seed)
	np.random.seed(seed)
	env.seed(seed)


if __name__ == '__main__':
	envs_kitchen = [
		"kitchen-complete-v0",
		# "kitchen-partial-v0",
		# "kitchen-mixed-v0",
	]
	envs_antmaze = [
		"antmaze-medium-play-v2",
  		"antmaze-medium-diverse-v2",
		"antmaze-large-play-v2",
		"antmaze-large-diverse-v2",
		"antmaze-umaze-v2",
		"antmaze-umaze-diverse-v2",
	]
	envs_adroit = [
		"pen-human-v1",
		"pen-cloned-v1",
		"door-human-v1",
		"door-cloned-v1",
		"hammer-human-v1",
		"hammer-cloned-v1",
	]
	envs_gym = [
		"halfcheetah-medium-expert-v2",
		"hopper-medium-expert-v2",
		"walker2d-medium-expert-v2",
		"halfcheetah-medium-replay-v2",
		"hopper-medium-replay-v2",
		"walker2d-medium-replay-v2",
		"halfcheetah-medium-v2",
		"hopper-medium-v2",
		"walker2d-medium-v2",
		"halfcheetah-random-v2",
		"hopper-random-v2",
		"walker2d-random-v2",
	]
	
	task_name = 'kitchen'
	
	config = json.load(open(f'config/{task_name}/config.json', 'r'))
	env_list = envs_kitchen
	if task_name == 'kitchen':
		env_list = envs_kitchen
	elif task_name == 'antmaze':
		env_list = envs_antmaze
	elif task_name == 'adroit':
		env_list = envs_adroit
	elif task_name == 'gym':
		env_list = envs_gym
	else:
		assert False, 'invalid environment!'
	
	env_score_log = {}
	mean_score_log, std_score_log = [], []
	config['load_model'] = False
	seed = 0
	config['seed'] = seed
	for env_name in env_list:
		env = gym.make(env_name)
		obs = env.reset()
		score_log = {}
		total_score = 0
		tmp = []
		set_seed(env, seed=seed)
		if not os.path.exists(f'./models/{env_name}'):
			os.mkdir(f'./models/{env_name}')
		if not os.path.exists(f'./plots/{env_name}'):
			os.mkdir(f'./plots/{env_name}')
		config['seed'] = seed
		config['env_name'] = env_name
		config['env'] = env
		print(f'config: {config}')
		score = build_planner_components(env_name, config)
		score_log[seed] = score
		total_score += score
		tmp.append(np.round(np.mean(score), 1))
		print(f'env_name: {env_name}, score_log: {score_log}, mean_score: {total_score/len(score_log.keys())}, sum_score: {total_score}, tmp: {tmp}')
		env_score_log[env_name] = {'seed_score': tmp, 'seed_mean': np.round(np.mean(tmp), 1), 'seed_std': np.round(np.std(tmp), 1)}
		mean_score_log.append(np.round(np.mean(tmp), 1))
		std_score_log.append(np.round(np.std(tmp), 1))
	print(env_score_log)
	print(mean_score_log)
	print(std_score_log)