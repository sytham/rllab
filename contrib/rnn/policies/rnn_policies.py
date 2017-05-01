'''
"Traditional" RNN policies.
'''
import os
import numpy as np
from collections import OrderedDict

from scipy.integrate import ode

from rllab.core.serializable import Serializable
from rllab.policies.base import Policy
from rllab.core.parameterized import Parameterized
from contrib.rnn.policies.base import ContinuousTimeModel, Layer, RecurrentLayer, logisticmap, identity, EulerIntegrator


class RNNPolicy(Policy):
        
    def reset(self):
        for l in self._layers:
            l.reset()
             
    def get_state(self):
        d = OrderedDict()
        d['h'] = self.h
        return d
    
    def set_state(self, state):
        self.h = state['h']
    
    def get_action(self, observation):
        action = self.output(observation)
        return action, dict()
    
    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return True
    
    def get_param_values(self, **tags):
        ''' Currently only trainable through gradient-free methods. So ignoring tags.'''
        return np.concatenate([l.encode() for l in self._layers])

    def set_param_values(self, flattened_params, **tags):
        ''' Currently only trainable through gradient-free methods. So ignoring tags.'''
        ix = 0
        for l in self._layers:
            l.decode(flattened_params[ix:ix+l.flat_dim()]) 
            ix += l.flat_dim()
        if ix != len(flattened_params):
            raise ValueError("Theta length {0} does not match param dim {1}".format(len(flattened_params), ix))
    
    # internal state encode/decode
    @property
    def h(self):
        h = []
        for l in self._layers:
            try:
                h.append(l.state.copy())
            except AttributeError:
                pass
        return np.concatenate(h)
    @h.setter
    def h(self, value):
        ix = 0
        for l in self._layers:
            try:
                l.state[:] = value[ix:ix+l.state_dim] 
            except AttributeError:
                continue
            ix += l.state_dim
        if ix != len(value):
            raise ValueError("Theta length {0} does not match state dim {1}".format(len(value), ix))


class CTRNNPolicy(RNNPolicy, ContinuousTimeModel):
    ''' Base class for continuous-time recurrent models '''
    def __init__(self,
                 env_spec,
                 timeconstant=0.01,
                 env_timestep=0.01,
                 integrator='euler',
                 max_dt=0.005):
        self.timestep = env_timestep
        ContinuousTimeModel.__init__(self, timeconstant, integrator, max_dt)        
        super(CTRNNPolicy, self).__init__(env_spec)
        
    def get_action(self, observation):
        action = self.output(observation, self.timestep)
        return action, dict()
    
    def reset(self):
        super(CTRNNPolicy, self).reset()
        ContinuousTimeModel.reset(self)

    def set_state(self, state):
        super(CTRNNPolicy, self).set_state(state)
        ContinuousTimeModel.set_state(self, state)
        
    def __getstate__(self):
        d = Parameterized.__getstate__(self)
        d['integrator'] = self.integrator
        return d

    def __setstate__(self, d):
        Parameterized.__setstate__(self, d)
        self.integrator = d['integrator']
        self.set_integrator(self.integrator)
    
    def get_param_values(self, **tags):
        th = super(CTRNNPolicy, self).get_param_values(**tags)
        return np.concatenate([np.array([self.timeconstant]), th])

    def set_param_values(self, flattened_params, **tags):
        self.timeconstant = flattened_params[0]
        super(CTRNNPolicy, self).set_param_values(flattened_params[1:], **tags)

        
class DeterministicDTRNNPolicy(RNNPolicy):
    '''
    Vanilla discrete-time recurrent neural net.
    '''
    def __init__(
            self,
            env_spec,
            hidden_sizes=(2,),
            hidden_nonlinearity=np.tanh,
            state_initf=None):
        
        Serializable.quick_init(self, locals())
        num_in = env_spec.observation_space.flat_dim
        num_out = env_spec.action_space.flat_dim
        
        self._layers = [RecurrentLayer(num_in, hidden_sizes[0], hidden_nonlinearity, state_initf=state_initf)]
        self._layers.extend([RecurrentLayer(hidden_sizes[i-1], hidden_sizes[i], hidden_nonlinearity, state_initf=state_initf)
                             for i in range(1,len(hidden_sizes))])
        self._layers.append(Layer(hidden_sizes[-1], num_out, identity))
        
        super(DeterministicDTRNNPolicy, self).__init__(env_spec)
        
     
    def output(self, x):
        for l in self._layers:
            x = l.out(x)
        return x
        

class DeterministicCTRNNPolicy(CTRNNPolicy):
    '''
    Vanilla continuous-time recurrent neural net.
    '''
    def __init__(
            self,
            env_spec,
            hidden_sizes=(2,),
            hidden_nonlinearity=np.tanh,
            timeconstant=0.01,
            env_timestep=0.01,
            state_initf=None,
            integrator='euler',
            max_dt=0.005):
        
        Serializable.quick_init(self, locals())
                
        num_in = env_spec.observation_space.flat_dim
        num_out = env_spec.action_space.flat_dim
                
        self._layers = [RecurrentLayer(num_in, hidden_sizes[0], hidden_nonlinearity, state_initf=state_initf)]
        self._layers.extend([RecurrentLayer(hidden_sizes[i-1], hidden_sizes[i], hidden_nonlinearity, state_initf=state_initf)
                             for i in range(1,len(hidden_sizes))])
        self._layers.append(Layer(hidden_sizes[-1], num_out, identity))
           
        super(DeterministicCTRNNPolicy, self).__init__(env_spec, timeconstant, env_timestep, integrator, max_dt)
        
        if hidden_sizes[-1] == num_out:
            self._layers[-1].W  = np.eye(self._layers[-1].W.shape[0])
                                                   
    def _calculate_dh(self, t, h, *args):
        self.h = h
        x = self.x
        for l in self._layers[:-1]:
            x = l.out(x)
        out = (-h + self.h) / self.timeconstant
        return out
    
    def _output_from_h(self):
        return self._layers[-1].out(self._layers[-2].state)
            
    
             
# class DeterministicCTRNNPolicy(RNNPolicy, Serializable, Parameterized):
#     '''
#     Vanilla continuous-time recurrent neural net.
#     '''
#     def __init__(
#             self,
#             env_spec,
#             hidden_sizes=(2,),
#             hidden_nonlinearity=np.tanh,
#             timeconstant=0.01,
#             env_timestep=0.01,
#             state_initf=None,
#             integrator='euler'):
#         
#         Serializable.quick_init(self, locals())
#         Parameterized.__init__(self)
#         
#         num_in = env_spec.observation_space.flat_dim
#         num_out = env_spec.action_space.flat_dim
#         self.timestep = env_timestep
#         
#         self._layers = [RecurrentLayer(num_in, hidden_sizes[0], hidden_nonlinearity, state_initf=state_initf)]
#         self._layers.extend([RecurrentLayer(hidden_sizes[i-1], hidden_sizes[i], hidden_nonlinearity, state_initf=state_initf)
#                              for i in range(1,len(hidden_sizes))])
#         self._layers.append(Layer(hidden_sizes[-1], num_out, identity))
#         
#         #ContinuousTimeModel.__init__(self, timeconstant, integrator)
#         self.tc = timeconstant
#         self.integrator = integrator
#         self.set_integrator(integrator)
#         self.reset()
#         
#         super(DeterministicCTRNNPolicy, self).__init__(env_spec)
#               
#         
#     def set_integrator(self, integrator):
#         if integrator == 'rk4': integrator = 'dopri5'
#         if integrator == 'euler':
#             self.ode = EulerIntegrator(self._calculate_dh)
#             self.ode_native_iter = 1
#         else:
#             #raise NotImplementedError("Scipy ODE causes problems when pickling / parallelizing")
#             self.ode = ode(self._calculate_dh).set_integrator(integrator)
#             self.ode_native_iter = 4
#             
#     def output(self, x, timestep=1):
#         self.x = x
#         dt = 0.005
#         numiter = max(1, int(timestep / (self.ode_native_iter * dt)))
#         for _ in range(numiter):
#             #self.h += dt * self._calculate_dh(0, self.h)
#             self.h = self.ode.integrate(self.ode.t + dt)
#         return self._output_from_h()
#                 
#     def get_state(self):
#         d = OrderedDict()
#         d['h'] = self.h.copy()
#         return d        
#         
#     @property
#     def timeconstant(self):
#         return self.tc
#     @timeconstant.setter
#     def timeconstant(self, value):
#         self.tc = max(value, 0.01)
#     
#     def get_param_values(self, **tags):
#         th = super(DeterministicCTRNNPolicy, self).get_param_values(**tags)
#         return np.concatenate([np.array([self.timeconstant]), th])
# 
#     def set_param_values(self, flattened_params, **tags):
#         self.timeconstant = flattened_params[0]
#         super(DeterministicCTRNNPolicy, self).set_param_values(flattened_params[1:], **tags)
#             
#     def _calculate_dh(self, t, h, *args):
#         x = self.x
#         for l in self._layers[:-1]:
#             x = l.out(x)
#         out = (-h + self.h) / self.timeconstant
#         return out
#     
#     def _output_from_h(self):
#         return self._layers[-1].out(self._layers[-2].state)
#             
#     def get_action(self, observation):
#         action = self.output(observation, self.timestep)
#         return action, dict()
#     
#     def reset(self):
#         RNNPolicy.reset(self)
#         self.ode.set_initial_value(self.h, 0)
#         #super(DeterministicCTRNNPolicy, self).reset()
# 
#     def set_state(self, state):
#         RNNPolicy.set_state(self, state)
#         self.ode.set_initial_value(self.h, 0)
#         #super(DeterministicCTRNNPolicy, self).set_state(state)
#         
#     def __getstate__(self):
#         d = Parameterized.__getstate__(self)
#         d['integrator'] = self.integrator
#         return d
# 
#     def __setstate__(self, d):
#         Parameterized.__setstate__(self, d)
#         self.integrator = d['integrator']
#         self.set_integrator(self.integrator)