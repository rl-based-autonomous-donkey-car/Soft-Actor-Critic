# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin

import argparse
import os
import time

import gym
import numpy as np
from PIL import Image
from stable_baselines.common import set_global_seeds

from config import ENV_ID, INPUT_DIM, MIN_STEERING, MAX_STEERING, JERK_REWARD_WEIGHT, MAX_STEERING_DIFF
from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from processing import Vae



algo = "sac"
folder = "logs"

vae_path = '/home/neo-47/Desktop/UntitledFolder/test/vae-32-final.pkl'

exp_id = 0

if exp_id == 0:
    exp_id = get_latest_run_id(os.path.join(folder, algo), ENV_ID)
    print('Loading latest experiment, id={}'.format(exp_id))

# Sanity checks
if exp_id > 0:
    log_path = os.path.join(folder, algo, '{}_{}'.format(ENV_ID, exp_id))
else:
    log_path = os.path.join(folder, algo)

best_path = ''

best_model = False

if best_model:
    best_path = '_best'

model_path = os.path.join(log_path, "{}{}.pkl".format(ENV_ID, best_path))


assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, ENV_ID, model_path)

set_global_seeds(0)


stats_path = os.path.join(log_path, ENV_ID)


hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False)
hyperparams['vae_path'] = vae_path

reward_log = ""

log_dir = reward_log if reward_log != '' else None

#env = create_test_env(stats_path=stats_path, seed=0, log_dir=log_dir,
                      #hyperparams=hyperparams)

model = ALGOS["sac"].load(model_path)

#obs = env.reset()

#print(obs.shape)

# Force deterministic for SAC and DDPG
deterministic = False or algo in ['ddpg', 'sac']

verbose = 1
if verbose >= 1:
    print("Deterministic actions: {}".format(deterministic))


running_reward = 0.0
ep_len = 0
n_timesteps = 2000
no_render = False

'''

for _ in range(n_timesteps):

    action, _ = model.predict(obs, deterministic=deterministic)


    # Clip Action to avoid out of bound errors
    if isinstance(env.action_space, gym.spaces.Box):
        action = np.clip(action, env.action_space.low, env.action_space.high)
    obs, reward, done, infos = env.step(action)

    if not no_render:
        env.render('human')
    running_reward += reward[0]
    ep_len += 1

    if done and verbose >= 1:
        # NOTE: for env using VecNormalize, the mean reward
        # is a normalized reward when `--norm_reward` flag is passed
        print("Episode Reward: {:.2f}".format(running_reward))
        print("Episode Length", ep_len)
        running_reward = 0.0
        ep_len = 0

env.reset()
env.envs[0].env.exit_scene()
# Close connection does work properly for now
# env.envs[0].env.close_connection()
time.sleep(0.5)
'''

Vae = Vae()
vae_32 = Vae.load_vae(path = "vae-32")

class PilotKeras:

    def __init__(self, min_throttle=0.2, max_throttle=0.5):

        self.action = (0, 0)
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle

    def run(self, img_arr):

        image = Vae.preprocessing(img_arr)
  
        image = vae_32.encode(image)

        image, command_history = Vae.postprocessing(self.action, image)

        self.action, _ = model.predict(image, deterministic = deterministic)


        # Convert from [-1, 1] to [0, 1]
        t = (self.action[1] + 1) / 2
        # Convert from [0, 1] to [min, max]
  
        self.action[1] = (1 - t) * self.min_throttle + self.max_throttle * t

        # Clip steering angle rate to enforce continuity
        
        prev_steering = command_history[0, -2]
        max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
        diff = np.clip(self.action[0] - prev_steering, -max_diff, max_diff)
        self.action[0] = prev_steering + diff

        # Clip Action to avoid out of bound errors    
        self.action = np.clip(self.action, [-1., -1.], [1., 1.])


        return self.action[0], self.action[1]


    def load(self, model_json_file):

        model = ALGOS["sac"].load(model_path)

        self.model = model

#or_image = Image.open("1.jpg")

#inst = PilotKeras()

#print(inst.run(or_image))

'''
or_image = Image.open("1.jpg")


Vae = Vae()
vae_32 = Vae.load_vae(path = "vae-32")
action = (0, 0)

for i in range(40):

    image = Vae.preprocessing(or_image)
  
    image = vae_32.encode(image)

    image = Vae.postprocessing(action, image)

    action, _ = model.predict(image, deterministic = deterministic)

    # Clip Action to avoid out of bound errors    
    action = np.clip(action, [-1., -1.], [1., 1.])

    print(action)

'''


