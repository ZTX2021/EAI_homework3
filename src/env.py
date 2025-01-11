from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
import gymnasium as gym
import warnings
import copy
from gymnasium import core, spaces
from typing import Dict, Optional,OrderedDict
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def dmc_spec2gym_space(spec):
	if isinstance(spec, OrderedDict):
		spec = copy.copy(spec)
		for k, v in spec.items():
			spec[k] = dmc_spec2gym_space(v)
		return spaces.Dict(spec)
	elif isinstance(spec, specs.BoundedArray):
		return spaces.Box(low=spec.minimum,
						  high=spec.maximum,
						  shape=spec.shape,
						  dtype=spec.dtype)
	elif isinstance(spec, specs.Array):
		return spaces.Box(low=-float('inf'),
						  high=float('inf'),
						  shape=spec.shape,
						  dtype=spec.dtype)
	else:
		raise NotImplementedError

class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats

	def step(self, action):
		reward = 0.0
		for i in range(self._num_repeats):
			obs,rew,done,_ = self._env.step(action)
			reward += rew
			if done:
				break

		return obs, reward, done, {}

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self):
		return self._env.reset()
	

	def __getattr__(self, name):
		if name in ["observation_space", "action_space"]:
			return getattr(self._env, name)
		return super().__getattr__(name)
	
	def next(self,state:np.array, action:np.array):
		return self._env.next(state, action,action_repeat=self._num_repeats)
			
		
	
class DMCEnv(core.Env):
	def __init__(self,
				 domain_name: str,
				 task_name: str,
				 task_kwargs: Optional[Dict] = {},
				 environment_kwargs=None):
		assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
		self._env = suite.load(domain_name=domain_name,
							   task_name=task_name,
							   task_kwargs=task_kwargs,
							   environment_kwargs=environment_kwargs)
		
		self.action_space = dmc_spec2gym_space(self._env.action_spec())
		
		self.extra_feat_dim=1+3
		
		self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self._env.physics.get_state().shape[0]+3+self.extra_feat_dim,), dtype=np.float32)
		self.action_space.seed(task_kwargs['random'])
		self.observation_space.seed(task_kwargs['random'])

		super().reset(seed=task_kwargs['random'])

	def step(self, action: np.ndarray):
		action=np.clip(action,self.action_space.low,self.action_space.high)
		if not self.action_space.contains(action):
			print('wrong action:', action, "action space: ", self.action_space)

		time_step = self._env.step(action)
		reward = time_step.reward or 0
		done = time_step.last()
		body_obs=self._env.physics.get_state()
		target=self._env.physics.named.data.geom_xpos['target']
		obs=np.concatenate([body_obs,target,time_step.observation['upright'][None,],time_step.observation['target']])
		return obs, reward, done, {}

	def reset(self, seed: Optional[int] = None, **kwargs):
		if seed is not None:
			self._env.task.random.seed(seed)
		time_step = self._env.reset()
		body_obs=self._env.physics.get_state()
		target=self._env.physics.named.data.geom_xpos['target']
		obs=np.concatenate([body_obs,target,time_step.observation['upright'][None,],time_step.observation['target']])
		return obs
	
	def seed(self, seed: int = 0):
		super().reset(seed=seed)

	def next(self,state, action: np.ndarray,action_repeat=1):
		"""Predict next state given current state and action."""
		state=state[:-self.extra_feat_dim]
		body_state, target=state[:-3],state[-3:]
		self._env.reset()
		self._env.physics.reset()
		self._env.physics.set_state(body_state)
		self._env.physics.named.model.geom_pos['target', 'x']=target[0]
		self._env.physics.named.model.geom_pos['target', 'y']=target[1]
		self._env.physics.named.model.geom_pos['target', 'z']=target[2]
		self._env.physics.forward()
		action=np.clip(action,self.action_space.low,self.action_space.high)
		rew=0
		for _ in range(action_repeat):
			time_step=self._env.step(action)
			rew+=time_step.reward
		next_body = self._env.physics.get_state()
		next_target=self._env.physics.named.data.geom_xpos['target']
		next_state=np.concatenate([next_body,next_target,time_step.observation['upright'][None,],time_step.observation['target']])
		return next_state,rew

def make_env(cfg):
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	assert domain=="fish" and task=="swim"
	assert (domain, task) in suite.ALL_TASKS
	env=DMCEnv(domain,task,task_kwargs={'random':cfg.seed})
	env = ActionRepeatWrapper(env, cfg.action_repeat)

	# Convenience
	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]

	return env
