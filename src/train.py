import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gymnasium as gym
import hydra
import time
import random
from pathlib import Path
from env import make_env
from agent import ModelBasedAgent
from helper import Episode, ReplayBuffer
import logger
import tqdm

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.set_printoptions(precision=8)


def evaluate(env, agent:ModelBasedAgent, num_episodes, step, cfg):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	eval_pb= tqdm.tqdm(range(num_episodes*cfg.episode_length), desc='Evaluating', unit='step')
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		for t in range(250):
			if cfg.policy=='random':
				action = env.action_space.sample()
			else: # mpc
				action = agent.plan(obs, eval_mode=True, step=step, t0=t==0).cpu().numpy()
			obs, reward, done, _ = env.step(action)
			ep_reward += reward
			eval_pb.update(1)
		episode_rewards.append(ep_reward)
	return np.nanmean(episode_rewards)

@hydra.main(version_base=None, config_path="../", config_name="cfg")
def train(cfg):
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env,env_fn = make_env(cfg), lambda: make_env(cfg)
	agent, buffer = ModelBasedAgent(cfg,env_fn), ReplayBuffer(cfg)
	if cfg.load_model:
		agent.load(cfg.load_path)
		print("Model loaded from ", cfg.load_path)
  
	L = logger.Logger(work_dir, cfg)
	if cfg.eval_only:
		# Evaluate agent
		episode_reward = evaluate(env, agent, cfg.eval_episodes, 0, cfg)
		L.log({'episode':0,'env_step':0,'episode_reward': episode_reward}, category='eval', agent=agent)
		L.finish(agent)
		return
 
	# Run training
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
		print(f"Step: {step}")
		if not cfg.use_real_model:
			# Collect trajectory
			obs = env.reset()
			episode = Episode(cfg, obs)
			while not episode.done:
				action = agent.plan(obs, step=step, t0=episode.first)
				obs, reward, done, _ = env.step(action.cpu().numpy())
				episode += (obs, action, reward, done)
			assert len(episode) == cfg.episode_length
			buffer += episode

			agent.model.update_statistics(
				obs=buffer._obs[: len(buffer)-1],
				acs=buffer._action[: len(buffer)-1],
				next_obs=buffer._obs[1: len(buffer)],
			)
			# Update model
			train_metrics = {}
			if step >= cfg.seed_steps:
				num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
				for i in range(num_updates):
					train_metrics.update(agent.update(buffer, step+i))

			# Log training episode
			episode_idx += 1
			env_step = int(step*cfg.action_repeat)
			common_metrics = {
				'episode': episode_idx,
				'step': step,
				'env_step': env_step,
				'total_time': time.time() - start_time,
				'episode_reward': episode.cumulative_reward}
			train_metrics.update(common_metrics)
			L.log(train_metrics, category='train')
		else:
			env_step=0
			common_metrics={'episode': episode_idx, 'step': step,'env_step': env_step, 'total_time': time.time() - start_time}
	
		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, cfg)
			L.log(common_metrics, category='eval', agent=agent)
			if cfg.use_real_model:
				break

	L.finish(agent)


if __name__ == '__main__':
	train()
