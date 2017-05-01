from rllab.envs.normalized_env import normalize
import contrib.rnn.envs.occluded_envs as occluded_envs

def get_base_env(env):
    base_env = env
    done = False
    while not done:
        try:
            base_env = base_env._wrapped_env
        except AttributeError:
            done = True
    return base_env

def construct_env(param_dict):
    base_env_cls = occluded_envs.get_env(param_dict["env_name"])
    base_env = base_env_cls(**param_dict.get("env_kwargs", {}))
    env = normalize(base_env, normalize_obs=True)
    
    real_base_env = get_base_env(base_env)
    env_dt = real_base_env.model.opt.timestep * real_base_env.frame_skip
    return env, env_dt