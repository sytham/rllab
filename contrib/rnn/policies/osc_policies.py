'''
Oscillator policies.
'''
import numpy as np

from rllab.core.serializable import Serializable

from contrib.rnn.policies.base import RecurrentLayer, identity, Layer
from contrib.rnn.policies.rnn_policies import CTRNNPolicy

# import bvp_ce
# 
# 
# class KuramotoLayer(RecurrentLayer):
#     def __init__(self, num_in, num_out, transferf=identity, bias=True, state_initf=None):
#         super(KuramotoLayer, self).__init__(num_in, num_out, transferf, bias, state_initf)
#         self.K = np.zeros((num_out, num_out))
#         self.K[range(0,num_out-1), range(1,num_out)] = 1.0
#     
#     @property
#     def state_dim(self):
#         return 3*self.num_out
#     
#     def flat_dim(self):
#         return Layer.flat_dim(self) + 1 # 1 for phi
#       
#     def encode(self):
#         return np.concatenate((Layer.encode(self), np.array([self.phi])))
#     
#     def decode(self, theta):
#         assert len(theta) == self.flat_dim()
#         Layer.decode(self, theta[:-1])
#         self.phi = theta[-1]
#              
#     def _coupling(self):
#         return self._rot_vec.dot(self._H).dot(self._K) - self._n*self._x
#     
#     def _one_step(self, S):
#         self._Hdot[0,:] = -self._y - self._z + self._coupling() + S #dx/dt
#         self._Hdot[1,:] = self._x + self._a*self._y                 #dy/dt
#         self._Hdot[2,:] = self._b + self._z*(self._x - self._c)     #dz/dt
#         return self._Hdot.T.reshape(-1,) # flatten to fit self.h
# 
#     def out(self, x):
#         self.h[:] = self._one_step(self.W.dot(x))
#         return self.h
#     
# class DeterministicKuramotoPolicy(CTRNNPolicy):
#     def __init__(self,
#                  env_spec,
#                  timeconstant=1./(8*np.pi),
#                  env_timestep=0.01,
#                  state_initf=None,
#                  integrator='rk4',
#                  max_dt=0.005):
#         Serializable.quick_init(self, locals())
#         super(DeterministicKuramotoPolicy, self).__init__(env_spec, timeconstant, env_timestep, integrator, max_dt)
#         
#         self.omega = np.zeros((self.num_out,1))
#         self.W = np.zeros((self.num_out, self.num_in+1))
#         self.K = np.zeros((self.num_out, self.num_out))
#         super(DeterministicKuramotoPolicy, self).__init__(num_in, num_hid, num_out, timeconstant)
#         
#     def _calculate_dh(self, t, h, *args):
#         self.h = h
#         x = self.x
#         # Calculate the change in h: h' = w + diag(K sin D) + Wx
#         D = self.h - self.h.T
#         return (self.omega + np.atleast_2d(np.diag(self.K.dot(np.sin(D)))).T + self.W.dot(x)) / self.timeconstant
#     
#     def _output_from_h(self):
#         return np.cos(self.z)
#     
# class DeterministicBVPPolicy(CTRNNPolicy):
#     '''
#     Bonhoeffer-Van der Pol oscillator with chaotic exploration.
#     See Phil Husbands, YoonSik Shim:
#     Incremental Embodied Chaotic Exploration of Self-Organized Motor Behaviors with Proprioceptor Adaptation. Front. Robotics and AI 2015 (2015)
#     '''
#     def __init__(
#             self,
#             env_spec,
#             timeconstant=0.01,
#             env_timestep=0.01):
#         
#         Serializable.quick_init(self, locals())
#         self.num_dof = env_spec.action_space.flat_dim
#         self.oscil = bvp_ce.OSCNET(self.num_dof)
#         self.oscil.create_controller(self.num_dof, timeconstant, 0.005)
#         self.timestep = env_timestep
#         self.HStop = 0.78
#         self.motorcommand = np.zeros((self.num_dof,), dtype=np.double)
#         
#     def output(self, x, timestep=1):
#         self.oscil.sense_joint(self.num_dof,
#                                angle.tolist(),
#                                aSpeed.tolist(),
#                                torque.tolist(),
#                                self.Hstop,
#                                inertia.tolist() )
#         
#         for _ in range(int(timestep/0.005)):
#             self.oscil.ForwardStep_Euler()
#         
#         for n in range(self.num_dof):
#             self.motorcommand[n] = self.oscil.get_oscil_out(n)
#         
#         return self.motorcommand
    

        
class RosslerLayer(RecurrentLayer):
    def __init__(self, num_in, num_out, transferf=identity, bias=True, state_initf=None):
        super(RosslerLayer, self).__init__(num_in, num_out, transferf, bias, state_initf)
                        
        self._params = [0.2, 3, 5.7]
        self.params = [0.2, 3, 5.7]
        # state vars
        self._x, self._y, self._z = self.h[0::3], self.h[1::3], self.h[2::3]
        self._H = self.h.reshape(-1,3).T # non-flat version of self.h, used in coupling calc
        # time derivative
        self._Hdot = np.zeros(self._H.shape)
        
        # phase offset
        self.phi = -0.1*np.pi
        # edge matrix (between Rossler oscillators). By default node tries to sync to node "ahead"
        self._K = np.zeros((num_out, num_out))
        self._K[range(0,num_out-1), range(1,num_out)] = 1.0
        # coupling gain. also sets self._K and self._n
        self.k = 0.5
        
        # coupling gain on sensory input
        self.ks = 1.0
            
    @property
    def phi(self):
        return self._phi
    @phi.setter
    def phi(self, v):
        self._phi = v
        self._rot_vec = np.atleast_1d([np.cos(self.phi), -np.sin(self.phi), 0])
    
    @property
    def k(self):
        return self._k 
    @k.setter
    def k(self, v):
        self._k = v
        self._K[np.nonzero(self._K)] = self._k
        self._n = np.sum(self._K, 0)
        
    @property
    def params(self):
        return self._params
    @params.setter
    def params(self, v):
        self._params[:] = v[:]
        self._a, self._b, self._c = self._params
    
    @property
    def state_dim(self):
        return 3*self.num_out
    
    def flat_dim(self):
        return Layer.flat_dim(self) + 1 # 1 for phi
      
    def encode(self):
        return np.concatenate((Layer.encode(self), np.array([self.phi])))
    
    def decode(self, theta):
        assert len(theta) == self.flat_dim()
        Layer.decode(self, theta[:-1])
        self.phi = theta[-1]
             
    def _coupling(self):
        return self._rot_vec.dot(self._H).dot(self._K) - self._n*self._x
    
    def _one_step(self, S):
        self._Hdot[0,:] = -self._y - self._z + self._coupling() + self.ks*S #dx/dt
        self._Hdot[1,:] = self._x + self._a*self._y                 #dy/dt
        self._Hdot[2,:] = self._b + self._z*(self._x - self._c)     #dz/dt
        return self._Hdot.T.reshape(-1,) # flatten to fit self.h

    def out(self, x):
        S = np.tanh(self.W.dot(x) + self.b)
        self.h[:] = self._one_step(S)
        return self.h

def rossler_initf(dim):
    # place init state roughly on attractor for a=0.2,b=3,c=5.7
    s = np.zeros(dim, dtype=np.float64)
    s[0::3] = 5.0
    s[1::3] = -1.0
    s[2::3] = 2.4
    return s

class DeterministicRosslerPolicy(CTRNNPolicy):
    def __init__(self,
                 env_spec,
                 hidden_sizes=(2,),
                 timeconstant=1./(4*np.pi),
                 env_timestep=0.01,
                 state_initf=None,
                 integrator='rk4',
                 max_dt=0.005):
        
        Serializable.quick_init(self, locals())
        
        num_in = env_spec.observation_space.flat_dim
        num_out = env_spec.action_space.flat_dim
                
        self._layers = [RosslerLayer(num_in, hidden_sizes[0], state_initf=state_initf)]
        self._layers.append(Layer(hidden_sizes[-1], num_out, identity))
        super(DeterministicRosslerPolicy, self).__init__(env_spec, timeconstant, env_timestep, integrator, max_dt)
        
    def _calculate_dh(self, t, h, *args):
        self.h = h
        x = self.x
        for l in self._layers[:-1]:
            x = l.out(x)
        return self.h / self.timeconstant
    
    def _output_from_h(self):
        return self._layers[-1].out(self._layers[-2]._x)
            
        
class DeterministicLimitCycleRosslerPolicy(DeterministicRosslerPolicy):
    def __init__(self,
                 env_spec,
                 timeconstant=1./(4*np.pi),
                 env_timestep=0.01,
                 state_initf=rossler_initf,
                 integrator='rk4'):
        
        Serializable.quick_init(self, locals())
        super(DeterministicLimitCycleRosslerPolicy, self).__init__(env_spec,
                                                                   (env_spec.action_space.flat_dim,),
                                                                   timeconstant,env_timestep,state_initf,integrator)
        self._layers[-1].W  = np.eye(self._layers[-1].W.shape[0])
        
    def get_param_values(self, **tags):
        return np.concatenate([
            np.array([self.timeconstant]),
            self._layers[-1].encode(),
            np.array([self._layers[0].phi])
            ])

    def set_param_values(self, flattened_params, **tags):
        assert len(flattened_params) == self._layers[-1].flat_dim() + 2
        self.timeconstant = flattened_params[0]
        self._layers[-1].decode(flattened_params[1:-1])
        self._layers[0].phi = flattened_params[-1]
        
class DeterministicSensoryRosslerPolicy(DeterministicRosslerPolicy):
    def __init__(self,
                 env_spec,
                 timeconstant=1./(4*np.pi),
                 env_timestep=0.01,
                 state_initf=rossler_initf,
                 integrator='rk4'):
        
        Serializable.quick_init(self, locals())
        super(DeterministicSensoryRosslerPolicy, self).__init__(env_spec,
                                                                   (env_spec.action_space.flat_dim,),
                                                                   timeconstant,env_timestep,state_initf,integrator)
        self._layers[-1].W  = np.eye(self._layers[-1].W.shape[0])
        self._layers[0].k = 0.0
        self._layers[0].W[1,-2] = 1.0
        self._layers[0].W[1,-1] = -1.0
        self._layers[0].b[-1] = 0.2*np.pi
        
    def get_param_values(self, **tags):
        print("Get param values")
        print(np.array([self.timeconstant]).ndim,
            self._layers[-1].encode().ndim,
            np.array([self._layers[0].ks]).ndim,
            self._layers[0].W.ravel().ndim,
            self._layers[0].b.ravel().ndim)
        
        return np.concatenate([
            np.array([self.timeconstant]),
            self._layers[-1].encode(),
            np.array([self._layers[0].ks]),
            self._layers[0].W.ravel(),
            self._layers[0].b.ravel()
            ])

    def set_param_values(self, flattened_params, **tags):
        assert len(flattened_params) == self._layers[-1].flat_dim() + 2 + Layer.flat_dim(self._layers[0])
        self.timeconstant = flattened_params[0]
        ix = 1+self._layers[-1].flat_dim()
        self._layers[-1].decode(flattened_params[1:ix])
        self._layers[0].ks = flattened_params[ix]
        Layer.decode(self._layers[0], flattened_params[ix+1:])
        