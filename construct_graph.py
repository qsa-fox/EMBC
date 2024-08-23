import numpy as np
from tqdm import tqdm
import torch


def construct_graph_reachability_batch(states, reachable_func, max_horizon, batch_size=5000):
	states = torch.FloatTensor(states)
	connect_chains = [{'next': [], 'weight': []} for _ in range(len(states))]
	n_batches = len(states) // batch_size + 1
	for i in tqdm(range(len(states)), 'construct_graph_reachability_full_clustering'):
		cur_idx = 0
		distance = []
		while True:
			end_idx = min(cur_idx + batch_size, len(states))
			batch_states = states[cur_idx:end_idx]
			with torch.no_grad():
				reachability_steps = reachable_func(torch.cat([states[i][None].repeat(len(batch_states), 1), batch_states], dim=-1).cuda()).squeeze().cpu().numpy()
				reachability_steps = reachability_steps.clip(min=0)
				distance.append(reachability_steps)
			cur_idx = end_idx
			if end_idx == len(states):
				break
		distance = np.concatenate(distance)
		reachable_idx = np.nonzero(distance < max_horizon)[0]
		reachable_idx = np.delete(reachable_idx, np.where(reachable_idx==i))
		connect_chains[i]['next'] = np.array(reachable_idx)
		connect_chains[i]['weight'] = distance[reachable_idx]
	return connect_chains
