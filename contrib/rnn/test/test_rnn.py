import unittest
import numpy as np
import matplotlib.pyplot as plt

from rllab.envs.env_spec import EnvSpec
from contrib.rnn.policies.rnn_policies import DeterministicDTRNNPolicy, DeterministicCTRNNPolicy
from contrib.rnn.policies.base import identity

class DummySpace(object):
    def __init__(self, dim):
        self.flat_dim = dim

class TestDTRNN(unittest.TestCase):

    def testIdentity(self):
        indim, outdim = 3,2
        rnn = DeterministicDTRNNPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)), hidden_nonlinearity=identity)
        
        W = np.array([[1,2,3],[4,5,6]])
        b = np.array([1,2])
        K = np.zeros((outdim,outdim))
        O = np.eye(outdim)
        ob = np.zeros((outdim,))
        rnn.set_param_values(np.concatenate([W.ravel(), b, K.ravel(), O.ravel(), ob]))
        
        x = np.array([1,2,3])
        self.assertTrue(np.all(rnn.output(x) == np.array([15,34])))
        
        # add recurrent connections
        K = np.eye(outdim)
        rnn.set_param_values(np.concatenate([W.ravel(), b, K.ravel(), O.ravel(), ob]))
        x = np.ones((indim,))
        self.assertTrue(np.all(rnn.output(x) == np.array([22,51])))
          
        # add lateral connections
        K = np.array([[0.9, 0.1], [0.1, 0.9]])
        rnn.set_param_values(np.concatenate([W.ravel(), b, K.ravel(), O.ravel(), ob]))
        o = rnn.output(x)
        h = np.array([0.9*22 + 0.1*51 + 7, 0.9*51 + 0.1*22 + 17]) # current state
        self.assertTrue(np.all(o == h))
        
        # get state
        s = rnn.get_state()
        self.assertTrue(np.all(s['h'] == h))
        
        # get and set params, one more forward pass
        rnn.set_param_values(rnn.get_param_values())
        self.assertTrue(np.all(rnn.output(x) == np.array([0.9*h[0] + 0.1*h[1]+7, 0.9*h[1] + 0.1*h[0] + 17])))
        
        # reset network state
        rnn.reset()
        self.assertFalse(np.all(rnn.get_state()['h'] == h))
        self.assertFalse(np.all(rnn.output(x) == np.array([0.9*h[0] + 0.1*h[1]+7, 0.9*h[1] + 0.1*h[0] + 17])))
        
        # restore to original state
        rnn.set_state(s)
        self.assertTrue(np.all(rnn.get_state()['h'] == h))
        
        # do the same forward pass again
        self.assertTrue(np.all(rnn.output(x) == np.array([0.9*h[0] + 0.1*h[1]+7, 0.9*h[1] + 0.1*h[0] + 17])))


class TestCTRNN(unittest.TestCase):

    def testEuler(self):
        indim, outdim = 3,2
        rnn = DeterministicCTRNNPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)),
                                       hidden_nonlinearity=identity,
                                       integrator='euler')
        
        W = np.array([[1,2,3],[4,5,6]])
        b = np.array([1,2])
        K = np.zeros((outdim,outdim))
        O = np.eye(outdim)
        ob = np.zeros((outdim,))
        timeconstant = 1.0
        rnn.set_param_values(np.concatenate([np.array([timeconstant]), W.ravel(), b, K.ravel(), O.ravel(), ob]))
        
        step = 0.001
        x = np.array([1,2,3])
        o = rnn.output(x, step)
        self.assertTrue(np.allclose(o, (step/rnn.timeconstant)*np.array([15.0, 34.0])))
        
        
    def testAll(self):
        indim, outdim = 1,1
                 
        W = np.array([0])
        b = np.array([1])
        K = np.zeros((outdim,outdim))
        O = np.eye(outdim)
        ob = np.array([0.5])
        step = 0.01
        x = np.array([3])
        tau = 0.02
        
        rnn = DeterministicCTRNNPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)),
                               hidden_sizes=(1,),
                               hidden_nonlinearity=identity,
                               timeconstant = tau,
                               integrator='euler',
                               state_initf=np.zeros,
                               max_dt=step)
        rnn.set_param_values(np.concatenate([np.array([tau]), W.ravel(), b, K.ravel(), O.ravel(), ob]))
        
        numsteps = 20
        y = [ob[0]] + [rnn.output(x, step) for _ in range(numsteps)]
        t = np.arange(numsteps+1) * step
        plt.figure()
        plt.plot(t, y, label='Euler')
        
        
        rnn = DeterministicCTRNNPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)),
                               hidden_sizes=(1,),
                               hidden_nonlinearity=identity,
                               timeconstant = tau,
                               integrator='midpoint',
                               state_initf=np.zeros,
                               max_dt=step)
        rnn.set_param_values(np.concatenate([np.array([tau]), W.ravel(), b, K.ravel(), O.ravel(), ob]))
        
        plt.plot(t, [ob[0]] + [rnn.output(x, step) for _ in range(numsteps)], label='Midpoint')
        
        rnn = DeterministicCTRNNPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)),
                               hidden_sizes=(1,),
                               hidden_nonlinearity=identity,
                               timeconstant = tau,
                               integrator='rk4',
                               state_initf=np.zeros,
                               max_dt=step)
        rnn.set_param_values(np.concatenate([np.array([tau]), W.ravel(), b, K.ravel(), O.ravel(), ob]))
         
        def z(t):
            c = -b[0] # from initial value z(0) = 0
            return c*np.exp(-t/tau) + b[0]
        
        plt.plot(t, [ob[0]] + [rnn.output(x, step) for _ in range(numsteps)], label='RK4')
        plt.plot(t, z(t) + ob[0], label='Exact')
        plt.legend()
                
        rnn = DeterministicCTRNNPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)),
                       hidden_sizes=(1,),
                       hidden_nonlinearity=identity,
                       timeconstant = tau,
                       integrator='rk4',
                       state_initf=np.zeros,
                       max_dt=step)
        rnn.set_param_values(np.concatenate([np.array([tau]), W.ravel(), b, K.ravel(), O.ravel(), ob]))
        o = rnn.output(x, step)
        oz = z(step) + ob[0]
        self.assertTrue(np.allclose(o, oz))
        self.assertTrue(np.allclose(rnn.output(x, step), z(2*step) + ob[0]))
        self.assertTrue(np.allclose(rnn.output(x, step), z(3*step) + ob[0]))
        
        plt.show()
        
        
if __name__ == "__main__":
    unittest.main()