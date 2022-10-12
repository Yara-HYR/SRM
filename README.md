# SRM
Spectrum Random Erasing



## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:

## Install MuJoCo
Download the MuJoCo version 2.1 binaries for Linux or OSX.

Extract the downloaded mujoco210 directory into \~/.mujoco/mujoco210.

If you want to specify a nonstandard location for the package, use the env variable MUJOCO_PY_MUJOCO_PATH.
pip3 install -U 'mujoco-py<2.2,>=2.1'


## Install DMControl
conda env create -f setup/conda.yml

conda activate dmcgb

sh setup/install_envs.sh


## Install CARLA
mkdir carla

tar -xvzf CARLA_0.9.9.4.tar.gz -C carla

cd carla/PythonAPI/carla/dist

easy_install carla-0.9.9-py3.7-linux-x86_64.egg

ln -fs carla/CarlaUE4.sh /usr/local/bin/carla-server


## Install Robosuite
pip install robosuite


## Install DrawerWorld
cd src/env/drawerworld

pip install -e .





## Usage
## DMControl Benchmark

from env.wrappers import make_env
env = make_env(\<br>
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode='train'
)
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)  


You can try other environments easily.
## Carla Benchmark
from env.carla import make_carla
env = make_carla(
    map_name=args.task_name,
    client_port=29000,
    npc_vehicles_port=29008,
    modalities = ["rgb",],
    frame_stack=3,
    weather = 'clear_noon',
    action_repeat=args.action_repeat,
    seed=args.seed
)


## Robosuite Benchmark
from env.robosuite import make_robosuite
env = make_robosuite(
    task=args.task_name,
    mode="train",
    scene_id=0,
)

## DrawerWorld Benchmark
from env.metaworld_wrappers import make_pad_env
env = make_pad_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        mode='train',
        action_factor=args.action_factor,
        moving_average_denoise=args.moving_average_denoise,
        moving_average_denoise_factor=args.moving_average_denoise_factor,
        moving_average_denoise_alpha=args.moving_average_denoise_alpha,
        exponential_moving_average=args.exponential_moving_average
)


## Training

MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=10  python3 src/train.py   --algorithm drq_aug   --seed 0 --tag SRM  --augmentation random_mask_freq; 
