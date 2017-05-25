from collections import OrderedDict

import numpy as np
from scipy.integrate import ode
import lasagne.init as linit

def identity(x):
    return x

def logisticmap(x):
    ''' alpha parameter gets absorbed in coupling'''
    return np.maximum(1 - x*x, -1.)

class Layer(object):
    def __init__(self, num_in, num_out, transferf=np.tanh, bias=True, initf=linit.GlorotUniform().sample):
        if num_in < 1 or num_out < 1:
            raise ValueError("Num in and out should be >= 1")
        if transferf is None:
            transferf=identity
            
        self.num_in = num_in
        self.num_out = num_out
        self.has_bias = bias
        self.transferf = transferf
        self.weight_initf = np.zeros if initf is None else initf
        self.W = self.weight_initf((num_out, num_in))
        self.b = np.zeros((num_out,))
        
    def dot(self, x):
        return self.W.dot(x) + self.b
    
    def out(self, x):
        return self.transferf(self.dot(x))
    
    def flat_dim(self):
        dim = self.num_in * self.num_out
        return dim + self.num_out if self.has_bias else dim
        
    def encode(self):
        if self.has_bias:
            return np.concatenate((self.W.ravel(), self.b))
        return self.W.ravel()
    
    def decode(self, theta):
        assert len(theta) == Layer.flat_dim(self)
        self.W[:,:] = theta[:self.num_in*self.num_out].reshape(self.num_out, self.num_in)
        if self.has_bias:
            self.b[:] = theta[-self.num_out:]
    
    def reset(self):
        pass
            
class RecurrentLayer(Layer):
    def __init__(self, num_in, num_out, state_initf=None, **kwargs):
        super(RecurrentLayer, self).__init__(num_in, num_out, **kwargs)
        self.K = self.weight_initf((num_out, num_out))
        self.state_initf = np.zeros if state_initf is None else state_initf
        self.h = self.state_initf((self.state_dim,))
        
    def dot(self, x):
        return self.K.dot(self.h) + self.W.dot(x) + self.b
   
    def out(self, x):
        self.h = self.transferf(self.dot(x))
        return self.h
    
    def flat_dim(self):
        return super(RecurrentLayer, self).flat_dim() + self.num_out**2
      
    def encode(self):
        return np.concatenate((super(RecurrentLayer, self).encode(), self.K.ravel()))
    
    def decode(self, theta):
        assert len(theta) == self.flat_dim()
        super(RecurrentLayer, self).decode(theta[:-self.num_out**2])
        self.K[:,:] = theta[-self.num_out**2:].reshape(self.num_out, self.num_out)
    
    def reset(self):
        self.h[:] = self.state_initf((self.state_dim,))
    
    @property
    def state_dim(self):
        return self.num_out
    @property
    def state(self):
        return self.h
    @state.setter
    def state(self, value):
        self.h[:] = value
        if self.transferf.__name__ in ('tanh', 'logisticmap'):
            self.h[:] = np.clip(self.h, -1.0, 1.0)

            
class EulerIntegrator(object):
    def __init__(self, dh):
        self.dh = dh
        self.t = 0
    def set_initial_value(self, h, t0):
        self.h = h
        self.t = t0
    def integrate(self, t):
        dt = t - self.t
        self.h += dt * self.dh(self.t, self.h)
        return self.h
    
class MidpointIntegrator(object):
    def __init__(self, dh):
        self.dh = dh
        self.t = 0
    def set_initial_value(self, h, t0):
        self.h = h
        self.t = t0
    def integrate(self, t):
        dt = t - self.t
        self.h += dt*self.dh(self.t+dt/2., self.h + self.dh(self.t, self.h)*dt/2.)
        return self.h
        
class ContinuousTimeModel(object):
    def __init__(self, timeconstant=1, integrator='euler', max_dt=0.005):
        self.tc = timeconstant
        self.integrator = integrator
        self.set_integrator(str.lower(integrator))
        self.max_dt = max_dt
        self.reset()
        
#     def output(self, x, timestep=1):
#         self.x = x
#         self.h = self.ode.integrate(self.ode.t + timestep)
#         return self._output_from_h()
    def output(self, x, timestep=1):
        self.x = x
        dt = min(timestep, self.max_dt)
        numiter = max(1, int(timestep / (self.ode_order*dt)))
        dt = float(timestep) / numiter
        #print("Doing", numiter, "iter for ode integrator order of", self.ode_order)
        for _ in range(numiter):
            self.h = self.ode.integrate(self.ode.t + dt)
        return self._output_from_h()
    
    def _calculate_dh(self, t, h, *args):
        '''
        Should calculate dh/dt, i.e. change in state.
        Method signature consistent with the one required by scipy.integrate.ode
        @param t: float, time
        @param h: ndarray, state
        @return ndarray, deriv of state wrt time
        '''
        raise NotImplementedError()
    
    def _output_from_h(self):
        ''' Given internal state h, calculate output. Typically linear transform.'''
        raise NotImplementedError()
    
    def reset(self):
        self.ode.set_initial_value(self.h, 0)
    
    def get_state(self):
        d = OrderedDict()
        d['h'] = self.h.copy()
        return d
    
    def set_state(self, state):
        self.ode.set_initial_value(self.h, 0)
    
    def set_integrator(self, integrator):
        if integrator == 'rk4': integrator = 'dopri5'
        if integrator == 'euler':
            self.ode = EulerIntegrator(self._calculate_dh)
            self.ode_order = 1
        elif integrator == 'midpoint':
            self.ode = MidpointIntegrator(self._calculate_dh)
            self.ode_order = 2
        else:
            self.ode = ode(self._calculate_dh).set_integrator(integrator)
            self.ode_order = 4
            
    @property
    def timeconstant(self):
        return self.tc
    @timeconstant.setter
    def timeconstant(self, value):
        self.tc = max(value, 0.01)