import numpy as np
import torch
import gym
import d4rl
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
import json
import os
from torch.optim.lr_scheduler import CosineAnnealingLR


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
	return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
	if isinstance(module, nn.Linear):
		if orthogonal_init:
			nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
			nn.init.constant_(module.bias, 0.0)
		else:
			nn.init.xavier_uniform_(module.weight, gain=1e-2)

class DiagGaussianPolicy(nn.Module):
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		max_action: float,
		orthogonal_init: bool = True,
		evaluation_deterministic: bool = False,
	):
		super().__init__()
		self.prev_evaluate_state = None
		self.observation_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action
		self.orthogonal_init = orthogonal_init
		self.evaluation_deterministic = evaluation_deterministic
		self.hidden_dim = 256
		self.base_network = nn.Sequential(
			nn.Linear(state_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			# nn.Linear(self.hidden_dim, self.hidden_dim),
			# nn.ReLU(),
			nn.Linear(self.hidden_dim, action_dim),
		)

		if orthogonal_init:
			self.base_network.apply(lambda m: init_module_weights(m, True))
		else:
			init_module_weights(self.base_network[-1], False)
		self.log_std = torch.nn.Parameter(torch.ones((1, action_dim)), requires_grad=True)

	def log_prob(
		self, observations: torch.Tensor, actions: torch.Tensor
	) -> torch.Tensor:
		if actions.ndim == 3:
			observations = extend_and_repeat(observations, 1, actions.shape[1])
		action_means = self.base_network(observations)
		normal_dist = torch.distributions.normal.Normal(loc=action_means, scale=self.log_std.exp())
		log_prob = normal_dist.log_prob(actions)
		return log_prob

	def forward(
		self,
		observations: torch.Tensor,
		deterministic: bool = False,
		repeat: bool = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		if repeat is not None:
			observations = extend_and_repeat(observations, 1, repeat)
		action_means = self.base_network(observations)
		if deterministic:
			action = action_means
		else:
			normal_dist = torch.distributions.normal.Normal(loc=action_means, scale=self.log_std.exp())
			action = normal_dist.rsample()
		return self.max_action * action

	@torch.no_grad()
	def act(self, state: np.ndarray, device: str = "cpu"):
		self.train(False)
		state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
		if self.prev_evaluate_state is None:
			self.prev_evaluate_state = state
		act_determ = True
		# recover from freeze state
		# if ((state - self.prev_evaluate_state)**2).mean() < 1e-3:
			# act_determ = False
		self.prev_evaluate_state = state
		with torch.no_grad():
			actions = self(state, deterministic=act_determ)
			# actions = self(state, deterministic=self.evaluation_deterministic)
		return actions.cpu().data.numpy().flatten().clip(-1.0, 1.0)

class GCRepalyBuffer:
	def __init__(self, sequence, valid_state_indices=None) -> None:
		self.sequence = []
		for seq in sequence:
			if len(seq['observations']) > 1:
				self.sequence.append(seq)
		self.sequence = np.array(self.sequence)
		self.validate_sequence = self.sequence[int(0.9 * len(self.sequence)):]
		self.sequence = self.sequence[: len(self.sequence) - len(self.validate_sequence)]
		self.seq_lens = np.array([len(seq['observations']) - 1 for seq in self.sequence])
		self.validate_seq_lens = np.array([len(seq['observations']) - 1 for seq in self.validate_sequence])
		self.states = np.concatenate([seq['observations'] for seq in sequence])
		self.state_mean = np.mean(self.states, axis=0)
		self.state_std = np.std(self.states, axis=0)
		if valid_state_indices is None:
			valid_state_indices = np.array(np.arange(self.states.shape[-1]))
		self.valid_state_indices = valid_state_indices
	
	def sample(self, batch_size=256, max_horizon=5, delta_target=False):
		seq_idx = np.random.choice(range(len(self.sequence)), batch_size, replace=True, p=self.seq_lens/np.sum(self.seq_lens))
		batch_seq = self.sequence[seq_idx]
		batch_seq_len = self.seq_lens[seq_idx]
		start_idx = batch_seq_len * np.random.rand(batch_size, )
		start_idx = np.array(start_idx, dtype=np.int32)
		if max_horizon == -1:
			end_idx = start_idx + np.random.randint(1, batch_seq_len-start_idx+1, size=(batch_size, ))
		else:
			end_idx = start_idx + np.random.randint(1, max_horizon+1, size=(batch_size, ))
		end_idx = np.minimum(end_idx, batch_seq_len - 1)
		states = np.stack([batch_seq[i]['observations'][start_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		actions = np.stack([batch_seq[i]['actions'][start_idx[i]] for i in range(len(batch_seq))])
		goals = np.stack([batch_seq[i]['observations'][end_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		if delta_target:
			delta_goals = goals - states
			states = np.concatenate([states, delta_goals], axis=-1)		
		else:
			states = np.concatenate([states, goals], axis=-1)		
		return states, actions
	
	def sample_validate(self, batch_size=256, max_horizon=5, delta_target=False):
		seq_idx = np.random.choice(range(len(self.validate_sequence)), batch_size, replace=True, p=self.validate_seq_lens/np.sum(self.validate_seq_lens))
		batch_seq = self.validate_sequence[seq_idx]
		batch_seq_len = self.validate_seq_lens[seq_idx]
		start_idx = batch_seq_len * np.random.rand(batch_size, )
		start_idx = np.array(start_idx, dtype=np.int32)
		if max_horizon == -1:
			end_idx = start_idx + np.random.randint(1, batch_seq_len-start_idx+1, size=(batch_size, ))
		else:
			end_idx = start_idx + np.random.randint(1, max_horizon+1, size=(batch_size, ))
		end_idx = np.minimum(end_idx, batch_seq_len - 1)
		states = np.stack([batch_seq[i]['observations'][start_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		actions = np.stack([batch_seq[i]['actions'][start_idx[i]] for i in range(len(batch_seq))])
		goals = np.stack([batch_seq[i]['observations'][end_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		if delta_target:
			delta_goals = goals - states
			states = np.concatenate([states, delta_goals], axis=-1)
		else:
			states = np.concatenate([states, goals], axis=-1)			
		return states, actions
		
	def sample_delta(self, batch_size=256, max_horizon=5):
		seq_idx = np.random.choice(range(len(self.sequence)), batch_size, replace=True, p=self.seq_lens/np.sum(self.seq_lens))
		batch_seq = self.sequence[seq_idx]
		batch_seq_len = self.seq_lens[seq_idx]
		start_idx = batch_seq_len * np.random.rand(batch_size, )
		start_idx = np.array(start_idx, dtype=np.int32)
		end_idx = start_idx + np.random.randint(1, max_horizon+1, size=(batch_size, ))
		end_idx = np.minimum(end_idx, batch_seq_len - 1)
		states = np.stack([batch_seq[i]['observations'][start_idx[i]] for i in range(len(batch_seq))])[:, :30]
		actions = np.stack([batch_seq[i]['actions'][start_idx[i]] for i in range(len(batch_seq))])
		goals = np.stack([batch_seq[i]['observations'][end_idx[i]] for i in range(len(batch_seq))])[:, :30]
		states = np.concatenate([states, goals], axis=-1)		
		return states, actions
	
	def sample_validate_delta(self, batch_size=256, max_horizon=5):
		seq_idx = np.random.choice(range(len(self.validate_sequence)), batch_size, replace=True, p=self.validate_seq_lens/np.sum(self.validate_seq_lens))
		batch_seq = self.validate_sequence[seq_idx]
		batch_seq_len = self.validate_seq_lens[seq_idx]
		start_idx = batch_seq_len * np.random.rand(batch_size, )
		start_idx = np.array(start_idx, dtype=np.int32)
		end_idx = start_idx + np.random.randint(1, max_horizon+1, size=(batch_size, ))
		end_idx = np.minimum(end_idx, batch_seq_len - 1)
		states = np.stack([batch_seq[i]['observations'][start_idx[i]] for i in range(len(batch_seq))])[:, :30]
		actions = np.stack([batch_seq[i]['actions'][start_idx[i]] for i in range(len(batch_seq))])
		goals = np.stack([batch_seq[i]['observations'][end_idx[i]] for i in range(len(batch_seq))])[:, :30]
		states = np.concatenate([states, goals], axis=-1)			
		return states, actions
		
	def sample_positive_reachability(self, batch_size=256, max_horizon=5, delta_target=False):
		seq_idx = np.random.choice(range(len(self.sequence)), batch_size, replace=True, p=self.seq_lens/np.sum(self.seq_lens))
		batch_seq = self.sequence[seq_idx]
		batch_seq_len = self.seq_lens[seq_idx]
		start_idx = batch_seq_len * np.random.rand(batch_size, )
		start_idx = np.array(start_idx, dtype=np.int32)
		horizon = np.random.randint(1, max_horizon+1, size=(batch_size, ))
		end_idx = start_idx + horizon
		end_idx = np.minimum(end_idx, batch_seq_len - 1)
		states = np.stack([batch_seq[i]['observations'][start_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		goals = np.stack([batch_seq[i]['observations'][end_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		if delta_target:
			delta_goals = goals - states
			states = np.concatenate([states, delta_goals], axis=-1)
		else:
			states = np.concatenate([states, goals], axis=-1)
		return states, horizon		
		
	def sample_negative_reachability(self, batch_size=256, delta_target=False):
		seq_idx = np.random.choice(range(len(self.sequence)), batch_size, replace=True, p=self.seq_lens/np.sum(self.seq_lens))
		batch_seq = self.sequence[seq_idx]
		batch_seq_len = self.seq_lens[seq_idx]
		start_idx = batch_seq_len * np.random.rand(batch_size, )
		start_idx = np.array(start_idx, dtype=np.int32)
		states = np.stack([batch_seq[i]['observations'][start_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		neg_idx = np.random.randint(0, len(self.states), size=(batch_size, ))
		goals = self.states[neg_idx][:, self.valid_state_indices]
		if delta_target:
			delta_goals = goals - states
			states = np.concatenate([states, delta_goals], axis=-1)
		else:
			states = np.concatenate([states, goals], axis=-1)
		return states	
		
	def sample4bc(self, batch_size=256):
		seq_idx = np.random.choice(range(len(self.sequence)), batch_size, replace=True, p=self.seq_lens/np.sum(self.seq_lens))
		batch_seq = self.sequence[seq_idx]
		batch_seq_len = self.seq_lens[seq_idx]
		start_idx = batch_seq_len * np.random.rand(batch_size, )
		start_idx = np.array(start_idx, dtype=np.int32)
		states = np.stack([batch_seq[i]['observations'][start_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		actions = np.stack([batch_seq[i]['actions'][start_idx[i]] for i in range(len(batch_seq))])
		return states, actions
	
	def sample4bc_validate(self, batch_size=256):
		seq_idx = np.random.choice(range(len(self.validate_sequence)), batch_size, replace=True, p=self.validate_seq_lens/np.sum(self.validate_seq_lens))
		batch_seq = self.validate_sequence[seq_idx]
		batch_seq_len = self.validate_seq_lens[seq_idx]
		start_idx = batch_seq_len * np.random.rand(batch_size, )
		start_idx = np.array(start_idx, dtype=np.int32)
		states = np.stack([batch_seq[i]['observations'][start_idx[i]] for i in range(len(batch_seq))])[:, self.valid_state_indices]
		actions = np.stack([batch_seq[i]['actions'][start_idx[i]] for i in range(len(batch_seq))])
		return states, actions

class HorizonReachability(nn.Module):
	def __init__(self, state_dim):
		super().__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 1)
		self.delta_target = False

	def forward(self, gc_state):
		a = F.relu(self.l1(gc_state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		out = self.l4(a)
		return out
		
	def predict(self, gc_state):
		return self.forward(gc_state)

def learn_reachability(dataset_sequence, config):
	device = config['device']
	input_dim = dataset_sequence[0]['observations'].shape[-1] * 2
	max_steps = int(config['reachability_net_train_steps'])
	max_horizon= int(config['reachability_net_horizon'])
	batch_size = config['reachability_net_batch_size']
	gc_buffer = GCRepalyBuffer(dataset_sequence)
	reachability_net = HorizonReachability(input_dim).to(device)
	reachability_net_optimizer = torch.optim.Adam(reachability_net.parameters(), lr=config['reachability_net_lr'], weight_decay=config['reachability_net_weight_decay'])
	reachability_net_lr_schedule = CosineAnnealingLR(reachability_net_optimizer, max_steps)

	loss_pos, loss_neg, loss_log = 0, 0, []
	for i in tqdm(range(max_steps), desc='learn_reachability'):
		pos_goal_states, horizon = gc_buffer.sample_positive_reachability(batch_size=batch_size, max_horizon=max_horizon)
		neg_goal_states = gc_buffer.sample_negative_reachability(batch_size=batch_size)
		pos_goal_states = torch.from_numpy(pos_goal_states).to(device)
		neg_goal_states = torch.from_numpy(neg_goal_states).to(device)
		cat_pos_neg_states = torch.cat([pos_goal_states, neg_goal_states], dim=0)
		horizon = torch.from_numpy(horizon).to(device)
   
		pos_neg_predict = reachability_net(cat_pos_neg_states).squeeze()
		pos_predict = pos_neg_predict[:len(pos_goal_states)]
		neg_predict = pos_neg_predict[len(pos_goal_states):]
		loss_pos = ((pos_predict - horizon)**2).mean()
		horizon_margin = 2
		loss_neg = (max_horizon + horizon_margin - neg_predict).clip(min=0).mean()
		loss = loss_pos + loss_neg
		reachability_net_optimizer.zero_grad()
		loss.backward()
		reachability_net_optimizer.step()
		reachability_net_lr_schedule.step()
		if i % 1000 == 0:
			print(f'iter: {i}, loss: {loss.item()}, loss_pos: {loss_pos.item()}, loss_neg: {loss_neg.item()}')
			loss_log.append(loss.item())
	return reachability_net

def learn_gcsl_actor(dataset_sequence, config):
	device = config['device']
	input_dim = dataset_sequence[0]['observations'].shape[-1] * 2
	action_dim = dataset_sequence[0]['actions'][0].shape[-1]
	max_steps = int(config['gcsl_net_train_steps'])
	max_horizon= int(config['gcsl_net_horizon'])
	batch_size = config['gcsl_net_batch_size']
	gc_buffer = GCRepalyBuffer(dataset_sequence)
	gcsl_actor = DiagGaussianPolicy(input_dim, action_dim, max_action=1.0).to(device)
	gcsl_actor_optimizer = torch.optim.Adam(gcsl_actor.parameters(), lr=config['gcsl_net_lr'], weight_decay=config['gcsl_net_weight_decay'])
	gscl_actor_lr_schedule = CosineAnnealingLR(gcsl_actor_optimizer, max_steps)
	gcsl_actor.train(True)
	
	loss_log, validate_loss_log = [], []
	for i in tqdm(range(max_steps), desc='learn_gcsl_actor'):
		goal_states, actions = gc_buffer.sample(batch_size=batch_size, max_horizon=max_horizon)
		goal_states = torch.from_numpy(goal_states).to(device)
		actions = torch.from_numpy(actions).to(device)
		log_prob = gcsl_actor.log_prob(goal_states, actions.clip(-0.999999, 0.999999))
		loss = -log_prob.mean()
		gcsl_actor_optimizer.zero_grad()
		loss.backward()
		gcsl_actor_optimizer.step()
		if config['gcsl_net_lr_schedule']:
			gscl_actor_lr_schedule.step()
		if i % 1000 == 0:
			with torch.no_grad():
				goal_states, actions = gc_buffer.sample_validate(batch_size=batch_size, max_horizon=max_horizon)
				goal_states = torch.from_numpy(goal_states).cuda()
				actions = torch.from_numpy(actions).cuda()
				log_prob = gcsl_actor.log_prob(goal_states, actions.clip(-0.99999, 0.99999))
				validate_loss = -log_prob.mean()
			print(f'iter: {i}, loss: {loss.item()}, validate_loss: {validate_loss.item()}')
			loss_log.append(loss.item())
			validate_loss_log.append(validate_loss.item())
	return gcsl_actor

def behavior_cloning(dataset_sequence, config):
	device = config['device']
	input_dim = dataset_sequence[0]['observations'].shape[-1]
	action_dim = dataset_sequence[0]['actions'].shape[-1]
	# max_steps = int(config['gcsl_net_train_steps'])
	max_steps = int(5e5)
	batch_size = config['gcsl_net_batch_size']
	gc_buffer = GCRepalyBuffer(dataset_sequence)
	actor = DiagGaussianPolicy(input_dim, action_dim, max_action=1.0).to(device)
	actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config['gcsl_net_lr'], weight_decay=config['gcsl_net_weight_decay'])
	actor_lr_schedule = CosineAnnealingLR(actor_optimizer, max_steps)
	actor.train(True)
	
	loss_log, validate_loss_log = [], []
	for i in tqdm(range(max_steps), desc='learn_gcsl_actor'):
		states, actions = gc_buffer.sample4bc(batch_size=batch_size)
		states = torch.from_numpy(states).to(device)
		actions = torch.from_numpy(actions).to(device)
		log_prob = actor.log_prob(states, actions.clip(-0.99999, 0.99999))
		loss = -log_prob.mean()
		actor_optimizer.zero_grad()
		loss.backward()
		actor_optimizer.step()
		actor_lr_schedule.step()
		if i % 1000 == 0:
			with torch.no_grad():
				states, actions = gc_buffer.sample4bc_validate(batch_size=batch_size)
				states = torch.from_numpy(states).cuda()
				actions = torch.from_numpy(actions).cuda()
				log_prob = actor.log_prob(states, actions.clip(-0.99999, 0.99999))
				validate_loss = -log_prob.mean()
			print(f'iter: {i}, loss: {loss.item()}, validate_loss: {validate_loss.item()}')
			loss_log.append(loss.item())
			validate_loss_log.append(validate_loss.item())
	return actor
