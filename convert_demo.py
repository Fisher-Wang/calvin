from dataclasses import dataclass
import os
import pickle
from typing import Literal

import numpy as np
import torch
import tyro
import yaml

from utils import quat_from_euler_xyz


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


args = tyro.cli(Args)

conf_path = f"{args.dataset_dir}/.hydra/merged_config.yaml"
ann_path = f"{args.dataset_dir}/lang_annotations/auto_lang_ann.npy"
scene_info_path = f"{args.dataset_dir}/scene_info.npy"
auto_lang_ann = np.load(ann_path, allow_pickle=True).item()
scene_info = np.load(scene_info_path, allow_pickle=True).item()
intervals = auto_lang_ann["info"]["indx"]
scene_cfg = yaml.load(open(f"calvin_env/conf/scene/calvin_scene_{args.scene}.yaml"), Loader=yaml.FullLoader)

FRANKA_NAME = "franka_with_gripper_extension"


traj = {FRANKA_NAME: []}

for traj_idx, (start_id, end_id) in enumerate(intervals):
    task_name = auto_lang_ann["language"]["task"][traj_idx]
    ## XXX: maybe start_id should not equal start_id or end_id?
    if not scene_info[f"calvin_scene_{args.scene}"][0] <= start_id <= scene_info[f"calvin_scene_{args.scene}"][1]:
        continue
    if task_name != args.task:
        continue
    print(f"Collecting No.{traj_idx} traj, task: {task_name}, frame range: {start_id} - {end_id}")

    metasim_states = []
    metasim_actions = []
    for i in range(start_id, end_id):
        path = f"{args.dataset_dir}/episode_{i:07d}.npz"
        data = np.load(path)
        actions, rel_actions, robot_obs, scene_obs = (
            data["actions"],
            data["rel_actions"],
            data["robot_obs"],
            data["scene_obs"],
        )

        robot_dof_pos = {
            "panda_joint1": robot_obs[7].item(),
            "panda_joint2": robot_obs[8].item(),
            "panda_joint3": robot_obs[9].item(),
            "panda_joint4": robot_obs[10].item(),
            "panda_joint5": robot_obs[11].item(),
            "panda_joint6": robot_obs[12].item(),
            "panda_joint7": robot_obs[13].item(),
            "panda_finger_joint1": 0.04 if robot_obs[14] == 1 else 0.0,
            "panda_finger_joint2": 0.04 if robot_obs[14] == 1 else 0.0,
        }

        ## See https://github.com/mees/calvin/blob/main/dataset/README.md#state-observation; Update: this is totally fucked up.
        # movable_objects = list(obj["file"] for obj in scene_cfg["objects"]["movable_objects"].values())
        # movable_objects = [o.replace("blocks/", "").replace(".urdf", "") for o in movable_objects]
        movable_objects = list(scene_cfg["objects"]["movable_objects"].keys())
        state = {
            "table": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0],
                "dof_pos": {
                    "base__slide": scene_obs[0].item(),
                    "base__drawer": scene_obs[1].item(),
                    "base__button": scene_obs[2].item(),
                    "base__switch": scene_obs[3].item(),
                },
            },
            movable_objects[0]: {
                "pos": scene_obs[6:9].tolist(),
                "rot": quat_from_euler_xyz(
                    torch.tensor(scene_obs[9]), torch.tensor(scene_obs[10]), torch.tensor(scene_obs[11])
                ).tolist(),
            },
            movable_objects[1]: {
                "pos": scene_obs[12:15].tolist(),
                "rot": quat_from_euler_xyz(
                    torch.tensor(scene_obs[15]), torch.tensor(scene_obs[16]), torch.tensor(scene_obs[17])
                ).tolist(),
            },
            movable_objects[2]: {
                "pos": scene_obs[18:21].tolist(),
                "rot": quat_from_euler_xyz(
                    torch.tensor(scene_obs[21]), torch.tensor(scene_obs[22]), torch.tensor(scene_obs[23])
                ).tolist(),
            },
            FRANKA_NAME: {
                # see calvin_env/conf/scene/calvin_scene_A.yaml
                "pos": [-0.34, -0.46, 0.24],
                "rot": [1.0, 0.0, 0.0, 0.0],
                "dof_pos": robot_dof_pos,
            },
        }
        metasim_states.append(state)

        action = {
            "dof_pos_target": robot_dof_pos,
        }
        metasim_actions.append(action)

    traj[FRANKA_NAME].append({
        "init_state": metasim_states[0],
        "states": metasim_states[1:],
        "actions": metasim_actions,
    })

    # break

os.makedirs(args.save_dir, exist_ok=True)
with open(f"{args.save_dir}/{args.task}_{args.scene}_v2.pkl", "wb") as f:
    pickle.dump(traj, f)
