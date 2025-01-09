from dataclasses import dataclass
import os
import time
from typing import Literal

import cv2
from loguru import logger as log
import numpy as np
from omegaconf import OmegaConf
import tyro
import yaml

from calvin_env.envs.play_table_env import PlayTableSimEnv


@dataclass
class Args:
    task: str = "lift_red_block_drawer"
    dataset_dir: Literal[
        "dataset/task_ABC_D/training",
        "dataset/task_ABC_D/validation",
        "dataset/calvin_debug_dataset/training",
        "dataset/calvin_debug_dataset/validation",
    ] = "dataset/task_ABC_D/training"
    scene: Literal["A", "B", "C", "D"] = "A"
    save_dir: str = os.path.expanduser("~/cod/RoboVerse/data_isaaclab/source_data/calvin_v2")
    states: bool = False


args = tyro.cli(Args)


dataset_dir = args.dataset_dir
conf_path = f"{dataset_dir}/.hydra/merged_config.yaml"
assert os.path.exists(conf_path)
render_conf = OmegaConf.load(conf_path)
scene_cfg = yaml.load(open(f"calvin_env/conf/scene/calvin_scene_{args.scene}.yaml"), Loader=yaml.FullLoader)
env = PlayTableSimEnv(
    robot_cfg=render_conf.robot,
    scene_cfg=scene_cfg,
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
scene_info_path = f"{args.dataset_dir}/scene_info.npy"
auto_lang_ann = np.load(ann_path, allow_pickle=True).item()
scene_info = np.load(scene_info_path, allow_pickle=True).item()
intervals = auto_lang_ann["info"]["indx"]

for traj_idx, (start_id, end_id) in enumerate(intervals):
    task_name = auto_lang_ann["language"]["task"][traj_idx]
    if f"calvin_scene_{args.scene}" not in scene_info:
        log.error(f"Scene {args.scene} not found in scene_info.npy")
        break
    if not scene_info[f"calvin_scene_{args.scene}"][0] <= start_id <= scene_info[f"calvin_scene_{args.scene}"][1]:
        continue
    if task_name != args.task:
        continue
    print(f"Replaying No.{traj_idx} traj, task: {task_name}, frame range: {start_id} - {end_id}")
    for i in range(start_id, end_id):
        path = f"{dataset_dir}/episode_{i:07d}.npz"
        data = np.load(path, allow_pickle=True)
        cv2.imshow("rgb_in_npz", data["rgb_static"][..., ::-1])
        cv2.waitKey(1)
        actions, rel_actions, robot_obs, scene_obs = (
            data["actions"],
            data["rel_actions"],
            data["robot_obs"],
            data["scene_obs"],
        )

        if i == start_id:
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        if args.states:
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        else:
            env.step(rel_actions)
