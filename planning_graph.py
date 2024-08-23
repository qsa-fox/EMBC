import numpy as np
from tqdm import tqdm
import torch

MAX_DIST = 99999.0


def chains2matrix(chains):
	matrix = np.ones((len(chains), len(chains))) * MAX_DIST
	for i in range(len(chains)):
		matrix[i][chains[i]['next']] = chains[i]['weight']
	return matrix

def floyed(weight_matrix):
	path_matrix = np.ones(weight_matrix.shape, dtype=np.int32)
	for i in range(len(path_matrix)):
		path_matrix[i] = np.array(np.arange(len(path_matrix[i])))
	
	weight_matrix = torch.FloatTensor(weight_matrix).cuda()
	# weight_matrix = torch.IntTensor(weight_matrix).cuda()
	path_matrix = torch.IntTensor(path_matrix).cuda()
	''' small pytorch float error will seriously influent the route matrix, eliminate the torch error with a margin eps.'''
	eps = 1e-4
 
	# for i in tqdm(range(len(weight_matrix)), 'floyed'):
	# 	new_weight = weight_matrix[:, i:i+1] + weight_matrix[i:i+1]
	# 	delta_weight = weight_matrix - new_weight
	# 	update_indices = delta_weight > eps
	# 	path_matrix[update_indices] = i
	# 	new_weight[~update_indices] = weight_matrix[~update_indices]
	# 	weight_matrix = torch.minimum(weight_matrix, new_weight)
	
	batch_size = 2000
	max_idx = len(weight_matrix)
	for i in tqdm(range(len(weight_matrix)), 'floyed'):
		cur_idx = 0
		while cur_idx < max_idx:
			end_idx = min(cur_idx + batch_size, max_idx)
			cur_idx2 = 0
			while cur_idx2 < max_idx:
				end_idx2 = min(cur_idx2 + batch_size, max_idx)
				new_weight = weight_matrix[cur_idx:end_idx, i:i+1] + weight_matrix[i:i+1, cur_idx2:end_idx2]
				delta_weight = weight_matrix[cur_idx:end_idx, cur_idx2:end_idx2] - new_weight
				update_indices = delta_weight > eps
				path_matrix[cur_idx:end_idx, cur_idx2:end_idx2][update_indices] = i
				new_weight[~update_indices] = weight_matrix[cur_idx:end_idx, cur_idx2:end_idx2][~update_indices]
				weight_matrix[cur_idx:end_idx, cur_idx2:end_idx2] = torch.minimum(weight_matrix[cur_idx:end_idx, cur_idx2:end_idx2], new_weight)
				cur_idx2 += batch_size
			cur_idx += batch_size
	
	return weight_matrix.cpu().numpy(), path_matrix.cpu().numpy()
				
def floyed_route(start_point, target_point, path_matrix):
	def _get_next_point(start_point, target_point, path_matrix):
		while True:
			mid_point = path_matrix[start_point][target_point]
			if mid_point == target_point:
				return mid_point
			else:
				return _get_next_point(start_point, mid_point, path_matrix)
	route = [start_point]
	while True:
		next_point = _get_next_point(start_point, target_point, path_matrix)
		route.append(next_point)
		start_point = next_point
		if next_point == target_point:
			break
	return route

def dijkstra(target_idx, connections):
	n_vertex = len(connections)
	preceding_idx = np.array([None for _ in range(n_vertex)])
	distances = np.array([MAX_DIST for _ in range(n_vertex)])
	distances[target_idx] = 0
	distances[connections[target_idx]['next']] = connections[target_idx]['weight']
	preceding_idx[connections[target_idx]['next']] = target_idx
	unvisited_vertexes = list(range(n_vertex))
	unvisited_vertexes.remove(target_idx)
	for i in tqdm(range(len(connections) - 1), desc='dijkstra'):
		unvisited_min_idx = np.argmin(distances[unvisited_vertexes])
		min_idx = unvisited_vertexes[unvisited_min_idx]
		unvisited_vertexes.remove(min_idx)
		for j in range(len(connections[min_idx]['next'])):
			new_distance = distances[min_idx] + connections[min_idx]['weight'][j]
			if new_distance < distances[connections[min_idx]['next'][j]]:
				distances[connections[min_idx]['next'][j]] = new_distance
				preceding_idx[connections[min_idx]['next'][j]] = min_idx
	return preceding_idx, distances

class RoutePlannerBC:
	def __init__(self, centers, end_idx, reachable_func=None, delta_target=False):
		self.centers = centers
		self.end_idx = end_idx
		self.reachable_func = reachable_func
		self.prev_goal_idx = None
		self.delta_target = delta_target
	
	def plan(self, start_point, *args):
		start_idx, _, distance = self._calcu_center(start_point)
		traj_idx_high = self.end_idx[start_idx < self.end_idx][0]
		start_idx = min(start_idx + 5, traj_idx_high-1)
  
		if self.prev_goal_idx is None:
			self.prev_goal_idx = start_idx
	
		self.step_cnt = 0
		self.prev_goal_idx = start_idx
		return self.centers[start_idx], [start_idx]

	def _calcu_center(self, pos):
		if self.reachable_func is None:
			distance = np.sum((pos[None] - self.centers)**2, axis=-1)
		else:
			if self.delta_target:
				states = torch.FloatTensor(np.concatenate([np.repeat(pos[None], len(self.centers), axis=0), self.centers - pos[None]], axis=-1)).cuda()
			else:
				states = torch.FloatTensor(np.concatenate([np.repeat(pos[None], len(self.centers), axis=0), self.centers], axis=-1)).cuda()
			distance = self.reachable_func.predict(states).detach().cpu().numpy().squeeze()
			
		center_idx = np.argmin(distance)
		center = self.centers[center_idx]
		return center_idx, center, np.min(distance)

class RoutePlannerDijkstra:
	def __init__(self, centers, connect_chains, goal, reachable_func=None):
		self.centers = centers
		self.connect_chains = connect_chains
		self.goal = goal
		self.reachable_func = reachable_func
		self.target_idx, _ = self._calcu_center(goal)
		self.build_graph()
		
	def build_graph(self):
		self.reversed_chains = self._reverse_direction(self.connect_chains)
		self.preceding_chain, self.distances = dijkstra(self.target_idx, self.reversed_chains)
  
	@staticmethod
	def _reverse_direction(connect_chains):
		reversed_chains = [{'next': [], 'weight': []} for _ in range(len(connect_chains))]
		for i in tqdm(range(len(connect_chains)), desc='reverse_direction'):
			next_idx = connect_chains[i]['next']
			next_weight = connect_chains[i]['weight']
			for j in range(len(next_idx)):
				reversed_chains[next_idx[j]]['next'].append(i)
				reversed_chains[next_idx[j]]['weight'].append(next_weight[j])
		return reversed_chains
	
	def plan(self, start_point, *args):
		start_idx, _ = self._calcu_center(start_point)
		preceding_idx = start_idx
		debug = True
		if not debug:
			next_jump = self.preceding_chain[preceding_idx]
			if next_jump is None:
				return self.centers[preceding_idx], None
			else:
				return self.centers[next_jump], None
		else:
			route = []
			route.append(preceding_idx)
			while True:
				if preceding_idx == self.target_idx or preceding_idx is None:
					if len(route) == 1:
						route.append(preceding_idx)
					break
				preceding_idx = self.preceding_chain[preceding_idx]
				route.append(preceding_idx)
			if route[-1] is None and len(route) == 2:
				route = route[:-1]
				route.append(start_idx)
			return self.centers[route[1]], route
	
	def get_shortest_route(self, start_point, *args):
		start_idx, _ = self._calcu_center(start_point)
		preceding_idx = start_idx
		route = [preceding_idx]
		route_states = [self.centers[preceding_idx]]
		while preceding_idx != self.target_idx:
			next_jump = self.preceding_chain[preceding_idx]
			if next_jump is None:
				print('no feasible route')
				break
			route.append(next_jump)
			route_states.append(self.centers[next_jump])
			preceding_idx = next_jump
		return np.array(route_states)
			
	def random_plan(self, start_point, *args):
		start_idx, _ = self._calcu_center(start_point)
		# start_idx = np.random.randint(0, len(self.centers))
		start_idx = min(start_idx+2, len(self.centers) - 1)
		return self.centers[start_idx], start_idx

	def _calcu_center(self, pos, print_distance=False):
		if self.reachable_func is None:
			distance = np.sum((pos[None] - self.centers)**2, axis=-1)
		else:
			states = torch.FloatTensor(np.concatenate([np.repeat(pos[None], len(self.centers), axis=0), self.centers], axis=-1)).cuda()
			distance = self.reachable_func.predict(states).detach().cpu().numpy().squeeze()
		center_idx = np.argmin(distance)
		center = self.centers[center_idx]
		min_distance = np.min(distance)
		if print_distance:
			print(f'min_distance: {min_distance}')
		return center_idx, center
