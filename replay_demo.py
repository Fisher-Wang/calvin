from calvin_env.envs.play_table_env import PlayTableSimEnv
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from omegaconf import OmegaConf
import hydra
import os
import time


## see dataset/calvin_debug_dataset/training/.hydra/merged_config.yaml
path = "dataset/calvin_debug_dataset/validation/.hydra/merged_config.yaml"
assert os.path.exists(path)
render_conf = OmegaConf.load(path)
# env = hydra.utils.instantiate(render_conf.env, show_gui=True, use_vr=False, use_scene_info=True)
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

eval_sequences = get_sequences(1000)

for initial_state, eval_sequence in eval_sequences:
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    while True:
        time.sleep(1)
        