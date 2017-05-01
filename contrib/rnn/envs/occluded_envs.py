from rllab.envs.occlusion_env import OcclusionEnv
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.hill.ant_hill_env import AntEnv, AntHillEnv
from rllab.envs.mujoco.hill.half_cheetah_hill_env import HalfCheetahEnv, HalfCheetahHillEnv
from rllab.envs.mujoco.hill.hopper_hill_env import HopperEnv, HopperHillEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.hill.swimmer3d_hill_env import Swimmer3DEnv, Swimmer3DHillEnv
from rllab.envs.mujoco.hill.walker2d_hill_env import Walker2DEnv, Walker2DHillEnv

''' ============= REGULAR ENVS ============== '''
class OccludedSwimmerEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedSwimmerEnv, self).__init__(SwimmerEnv(), [2,3,4]) # joint angles

class OccludedSwimmer3DEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedSwimmer3DEnv, self).__init__(Swimmer3DEnv(), [6,7,8]) # yaw and joint angles

class OccludedSwimmer3DZEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedSwimmer3DZEnv, self).__init__(Swimmer3DEnv(), [2,5,6,7,8]) #z-pos, pitch, yaw, joint angles
        
class OccludedAntEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedAntEnv, self).__init__(AntEnv(), list(range(2,15)) + [57,75,93,111]) # joint angles, leg ground normal contact force
        
class OccludedHalfCheetahEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedHalfCheetahEnv, self).__init__(HalfCheetahEnv(), list(range(8))) # z-pos, joint angles

class OccludedHopperEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedHopperEnv, self).__init__(HopperEnv(), [0,1,2,3,4,11,12,13,14,15,16]) # joint angles, constraint forces
        
class OccludedWalker2DEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedWalker2DEnv, self).__init__(Walker2DEnv(), list(range(2,10))) # joint angles (why no constraint forces here?)


''' ============= HILL ENVS ============== '''
class OccludedSwimmer3DHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedSwimmer3DHillEnv, self).__init__(Swimmer3DHillEnv(*args, **kwargs), [2,5,6,7,8]) #z-pos, pitch, yaw, joint angles
        
class OccludedAntHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedAntHillEnv, self).__init__(AntHillEnv(*args, **kwargs), list(range(2,15)) + [57,75,93,111]) # joint angles, leg ground normal contact force
        
class OccludedHalfCheetahHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedHalfCheetahHillEnv, self).__init__(HalfCheetahHillEnv(*args, **kwargs), list(range(8))) # z-pos, joint angles

class OccludedHopperHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedHopperHillEnv, self).__init__(HopperHillEnv(*args, **kwargs), [0,1,2,3,4,11,12,13,14,15,16]) # joint angles, constraint forces
        
class OccludedWalker2DHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedWalker2DHillEnv, self).__init__(Walker2DHillEnv(*args, **kwargs), list(range(2,10))) # joint angles (why no constraint forces here?)

constructors = {'swimmer-hill':OccludedSwimmer3DHillEnv,
               'ant-hill': OccludedAntHillEnv,
               'halfcheetah-hill': OccludedHalfCheetahHillEnv,
               'hopper-hill':OccludedHopperHillEnv,
               'walker2d-hill':OccludedWalker2DHillEnv,
               'swimmer':OccludedSwimmer3DEnv,
               'ant': OccludedAntEnv,
               'halfcheetah': OccludedHalfCheetahEnv,
               'hopper':OccludedHopperEnv,
               'walker2d':OccludedWalker2DEnv,
               'swimmer2d':OccludedSwimmerEnv,
               'swimmerz':OccludedSwimmer3DZEnv}

def get_env(name):
    return constructors[str.lower(name)]

def list_envs():
    return constructors.keys()
    