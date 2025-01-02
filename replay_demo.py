from calvin_env.envs.play_table_env import PlayTableSimEnv
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from omegaconf import OmegaConf
import hydra
import os
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--states', action='store_true')
args = parser.parse_args()

dataset_dir = "dataset/calvin_debug_dataset/validation"
conf_path = f"{dataset_dir}/.hydra/merged_config.yaml"
assert os.path.exists(conf_path)
render_conf = OmegaConf.load(conf_path)
env = PlayTableSimEnv(
    robot_cfg=render_conf.robot,
    scene_cfg=render_conf.scene,
    cameras=render_conf.cameras,
    show_gui=True,
    use_vr=False,
    use_scene_info=True,
    use_egl=True,
    seed=0,
    bullet_time_step=240.0,
    control_freq=30,
)

ann_path = f"{dataset_dir}/lang_annotations/auto_lang_ann.npy"
auto_lang_ann = np.load(ann_path, allow_pickle=True).item()
intervals = auto_lang_ann['info']['indx']

for traj_idx, (start_id, end_id) in enumerate(intervals):
    task_name = auto_lang_ann['language']['task'][traj_idx]
    print(traj_idx, task_name, start_id, end_id)
    for i in range(start_id, end_id):
        path = f"{dataset_dir}/episode_{i:07d}.npz"
        data = np.load(path)
        actions, rel_actions, robot_obs, scene_obs = data['actions'], data['rel_actions'], data['robot_obs'], data['scene_obs']

        if i == start_id:
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        
        if args.states:
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        else:
            env.step(rel_actions)
