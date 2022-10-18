import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.sac import SAC
import augmentations

from augmentations import random_mask_freq,random_mask_freq_v2#,random_mask_freq_FAN,random_mask_freq_LOWPASS,random_mask_freq_HIGHPASS,random_mask_freq_BANDPASS

class DrQ_aug(SAC): # [K=1, M=1]
        def __init__(self, obs_shape, action_shape, args):
                super().__init__(obs_shape, action_shape, args)
                print(args)
                self.aug_func = globals()[args.augmentation.rstrip()]

        def update(self, replay_buffer, L, step):
                obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()
                if self.aug_func == 'random_mask_freq_FAN':
                        obs = self.aug_func(obs,FAN_ANGLE=args.fan_angle)
                        next_obs = self.aug_func(next_obs, FAN_ANGLE=args.fan_angle)                    
                obs = self.aug_func(obs)
                next_obs = self.aug_func(next_obs)              

                self.update_critic(obs, action, reward, next_obs, not_done, L, step)

                if step % self.actor_update_freq == 0:
                        self.update_actor_and_alpha(obs, L, step)

                if step % self.critic_target_update_freq == 0:
                        self.soft_update_critic_target()

