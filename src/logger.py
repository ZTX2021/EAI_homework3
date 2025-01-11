import sys
import os
import datetime
import re
import numpy as np
import torch
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter



CONSOLE_FORMAT = [('episode', 'E', 'int'), ('env_step', 'S', 'int'), ('episode_reward', 'R', 'float'), ('total_time', 'T', 'time')]
AGENT_METRICS = ['consistency_loss', 'reward_loss', 'value_loss', 'total_loss', 'weighted_loss', 'pi_loss', 'grad_norm']


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg, reward=None):
	"""Pretty-printing of run information. Call at start of training."""
	prefix, color, attrs = '  ', 'green', ['bold']
	def limstr(s, maxlen=32):
		return str(s[:maxlen]) + '...' if len(str(s)) > maxlen else s
	def pprint(k, v):
		print(prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v))
	kvs = [('experiment', cfg.exp_name),
           ('task', cfg.task),
		   ('train steps', f'{int(cfg.train_steps*cfg.action_repeat):,}'),
           ('mpc_algo', cfg.mpc_algo) if cfg.policy=='mpc' else ('policy', cfg.policy),
            ('use_real_model', cfg.use_real_model),
       	   ('use_td', cfg.use_td),
           ]
	if reward is not None:
		kvs.append(('episode reward', colored(str(int(reward)), 'white', attrs=['bold'])))
	w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
	div = '-'*w
	print(div)
	for k,v in kvs:
		pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""Return a wandb-safe group name for logging. Optionally returns group name as list."""
	lst = [cfg.task, cfg.modality, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
	return lst if return_list else '-'.join(lst)


class Logger(object):
	"""Primary logger object. Logs either locally or using wandb."""
	def __init__(self, log_dir, cfg):
		self._log_dir = make_dir(log_dir)
		self._model_dir = make_dir(self._log_dir / 'models')
		self._save_model = cfg.save_model and (not cfg.use_real_model)
		self._save_freq = cfg.save_freq
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._cfg = cfg
		self._eval = []
		self.writer=SummaryWriter(log_dir)
		print_run(cfg)

	def finish(self, agent):
		if self._save_model:
			fp = self._model_dir / f'model.pt'
			agent.save(fp)
		self.writer.close()
		print_run(self._cfg, self._eval[-1][-1])

	def _format(self, key, value, ty):
		if ty == 'int':
			return f'{colored(key+":", "grey")} {int(value):,}'
		elif ty == 'float':
			return f'{colored(key+":", "grey")} {value:.01f}'
		elif ty == 'time':
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "grey")} {value}'
		else:
			raise f'invalid log format type: {ty}'

	def _print(self, d, category):
		category = colored(category, 'blue' if category == 'train' else 'green')
		pieces = [f' {category:<14}']
		for k, disp_k, ty in CONSOLE_FORMAT:
			pieces.append(f'{self._format(disp_k, d.get(k, 0), ty):<26}')
		print('   '.join(pieces))

	def log(self, d, category='train',agent=None):
		assert category in {'train', 'eval'}
		# tensorboard logging
		for k,v in d.items():
			self.writer.add_scalar(category + '/' + k, v, d['episode'])
		self.writer.flush()
		if category == 'eval':
			keys = ['env_step', 'episode_reward']
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(self._log_dir / 'eval.log', header=keys, index=None)
		self._print(d, category)
  
		if self._save_model and d['episode'] % self._save_freq == 1 and agent is not None:
			fp = self._model_dir / f'model_{d["episode"]}.pt'
			agent.save(fp)
