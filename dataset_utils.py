import numpy as np
import gym
import d4rl

	
def get_datasets_state_bc(env_name, config, evaluation_flag=False):
	env = gym.make(env_name)
	dataset_sequence = np.array(list(d4rl.sequence_dataset(env)))
	best_seqence = dataset_sequence
 
	if evaluation_flag:
		seq_len = np.array([len(seq['rewards']) for seq in dataset_sequence])
		dataset_sequence = dataset_sequence[seq_len > 10]
		if 'kitchen' in env_name or 'cloned' in env_name or 'human' in env_name:
			seq_rewards = np.array([np.max(seq['rewards']) for seq in dataset_sequence])
		elif 'hopper' in env_name or 'halfcheetah' in env_name or 'walker2d' in env_name:
			seq_rewards = np.array([np.sum(seq['rewards']) for seq in dataset_sequence])
		elif 'ant' in env_name:
			seq_rewards_max = np.array([np.max(seq['rewards']) for seq in dataset_sequence])
			dataset_sequence = dataset_sequence[seq_rewards_max >= 1.0]
			start_pos = [5, 5]
			tmp_idx = []
			for i in range(len(dataset_sequence)):
				if dataset_sequence[i]['observations'][0][0] < start_pos[0] and dataset_sequence[i]['observations'][0][1] < start_pos[1]:
					tmp_idx.append(i)
			dataset_sequence = dataset_sequence[tmp_idx]
			seq_len = np.array([len(seq['rewards']) for seq in dataset_sequence])
			seq_rewards = 1.0 / seq_len
		else:
			raise Exception('environment not supported!')
		best_sequence_idx = np.argsort(seq_rewards)
		n_demos = config['n_expert_demos']
		if n_demos == -1:
			best_seqence = dataset_sequence[best_sequence_idx]
		else:
			best_seqence = dataset_sequence[best_sequence_idx[-n_demos:]]
		
	best_seq_len = [len(seq['observations']) for seq in best_seqence]
	state_end_idx = np.cumsum(best_seq_len, axis=-1)
	state_datas = np.concatenate([seq['observations'] for seq in best_seqence])
	goal_state = best_seqence[-1]['observations'][-1]
	goal_idx = len(state_datas) - 1
	return best_seqence, state_datas, goal_state, goal_idx, state_end_idx

def get_datasets_state_stitching(env_name, config, evaluation_flag=False):
	env = gym.make(env_name)
	dataset_sequence = np.array(list(d4rl.sequence_dataset(env)))
	best_seqence = dataset_sequence
	raw_len = len(dataset_sequence)
	
	if evaluation_flag:
		seq_len = np.array([len(seq['rewards']) for seq in dataset_sequence])
		dataset_sequence = dataset_sequence[seq_len > 10]
		if 'kitchen' in env_name:
			seq_rewards = np.array([np.max(seq['rewards']) for seq in dataset_sequence])
			best_sequence_idx = np.argsort(seq_rewards)
			n_demos = config['n_expert_demos']
			# n_demos = int(0.1 * raw_len)
			# n_demos = -1
			if n_demos == -1:
				best_seqence = dataset_sequence[best_sequence_idx]
			else:
				best_seqence = dataset_sequence[best_sequence_idx[-n_demos:]]
			best_seq_len = [len(seq['observations']) for seq in best_seqence]
			state_end_idx = np.cumsum(best_seq_len, axis=-1)
			state_datas = np.concatenate([seq['observations'] for seq in best_seqence])
   
			goal_sequence = dataset_sequence = np.array(list(d4rl.sequence_dataset(gym.make('kitchen-complete-v0'))))
			goal_seq_rewards = np.array([np.max(seq['rewards']) for seq in goal_sequence])
			goal_state = goal_sequence[np.argsort(goal_seq_rewards)[-1]]['observations'][-1]
			state_datas = np.concatenate([state_datas, goal_state[None]], axis=0)
   
			goal_idx = len(state_datas) - 1
			return best_seqence, state_datas, goal_state, goal_idx, state_end_idx
   
		elif 'cloned' in env_name or 'human' in env_name:
			seq_rewards = np.array([np.max(seq['rewards']) for seq in dataset_sequence])
		elif 'hopper' in env_name or 'halfcheetah' in env_name or 'walker2d' in env_name:
			seq_rewards = np.array([np.sum(seq['rewards']) for seq in dataset_sequence])
		elif 'ant' in env_name:
			seq_rewards_max = np.array([np.max(seq['rewards']) for seq in dataset_sequence])
			rewards_idx = np.argsort(seq_rewards_max)
			goal_state = dataset_sequence[rewards_idx[-1]]['observations'][-1]
			# dataset_sequence = dataset_sequence[seq_rewards_max >= 1.0]
			start_pos = [5, 5]
			tmp_idx = []
			for i in range(len(dataset_sequence)):
				if dataset_sequence[i]['observations'][0][0] < start_pos[0] and dataset_sequence[i]['observations'][0][1] < start_pos[1]:
					tmp_idx.append(i)
			dataset_sequence = dataset_sequence[tmp_idx]
			seq_len = np.array([len(seq['rewards']) for seq in dataset_sequence])
			# seq_rewards = 1.0 / seq_len
			seq_rewards = np.array([np.max(seq['rewards']) for seq in dataset_sequence])
		else:
			raise Exception('environment not supported!')
		best_sequence_idx = np.argsort(seq_rewards)
		n_demos = config['n_expert_demos']
		# n_demos = int(0.1 * raw_len)
		# n_demos = -1
		if n_demos == -1:
			best_seqence = dataset_sequence[best_sequence_idx]
		else:
			best_seqence = dataset_sequence[best_sequence_idx[-n_demos:]]
		
	best_seq_len = [len(seq['observations']) for seq in best_seqence]
	state_end_idx = np.cumsum(best_seq_len, axis=-1)
	state_datas = np.concatenate([seq['observations'] for seq in best_seqence])
	# state_datas = state_datas[:, config['valid_state_indices']]
	# goal_state = best_seqence[-1]['observations'][-1][config['valid_state_indices']]
	goal_state = best_seqence[-1]['observations'][-1]
	goal_idx = len(state_datas) - 1
	return best_seqence, state_datas, goal_state, goal_idx, state_end_idx


def get_datasets_state_new_goal(env_name, config):
	assert 'ant' in env_name
	env = gym.make(env_name)
	dataset_sequence = np.array(list(d4rl.sequence_dataset(env)))
	best_seqence = dataset_sequence
	raw_len = len(dataset_sequence)
	
	seq_len = np.array([len(seq['rewards']) for seq in dataset_sequence])
	dataset_sequence = dataset_sequence[seq_len > 10]

	seq_rewards_max = np.array([np.max(seq['rewards']) for seq in dataset_sequence])
	# dataset_sequence = dataset_sequence[seq_rewards_max >= 1.0]
	start_pos = [5, 5]
	# end_pos = np.array([5, 20])
	end_pos = np.array([12, 25])
	# end_pos = np.array([32, 8])
	tmp_idx = []
	for i in range(len(dataset_sequence)):
		start_x, start_y = dataset_sequence[i]['observations'][0][0], dataset_sequence[i]['observations'][0][1]
		end_xy = dataset_sequence[i]['observations'][-1][:2]
		if start_x < start_pos[0] and start_y < start_pos[1]:
			tmp_idx.append(i)
	dataset_sequence = dataset_sequence[tmp_idx]
	seq_len = np.array([len(seq['rewards']) for seq in dataset_sequence])
	seq_rewards = 1.0 / seq_len
	best_sequence_idx = np.argsort(seq_rewards)
	
	n_demos = config['n_expert_demos']
	# n_demos = int(0.1 * raw_len)
	# n_demos = -1
	if n_demos == -1:
		best_seqence = dataset_sequence[best_sequence_idx]
	else:
		best_seqence = dataset_sequence[best_sequence_idx[-n_demos:]]
		
	best_seq_len = [len(seq['observations']) for seq in best_seqence]
	state_end_idx = np.cumsum(best_seq_len, axis=-1)
	state_datas = np.concatenate([seq['observations'] for seq in best_seqence])
	goal_state = end_pos
	goal_idx = len(state_datas) - 1
	return best_seqence, state_datas, goal_state, goal_idx, state_end_idx