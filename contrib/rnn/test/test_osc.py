import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rllab.envs.env_spec import EnvSpec
from contrib.rnn.policies.osc_policies import RosslerLayer, DeterministicRosslerPolicy, DeterministicSensoryRosslerPolicy
from contrib.rnn.envs.sync_env_spec import SyncEnvSpec

class DummySpace(object):
    def __init__(self, dim):
        self.flat_dim = dim

def init_fixedoffset(dim):
    return np.concatenate([np.array([-1.0,0.0,0.0,1.0,0.0,0.0]), np.zeros((dim[0]-6,))])
 
class TestRossler(unittest.TestCase):

    TIMECONSTANT = 0.05 # observe diff Euler vs RK4. Euler will start breaking down if tc decreased further with this timestep
    TIMESTEP = 0.005
    def testSensoryRossler(self):
        indim, outdim = 3,2
        step = self.TIMESTEP
        timeconstant = self.TIMECONSTANT
        spec = SyncEnvSpec(DummySpace(indim),
                           DummySpace(outdim),
                           np.zeros((outdim, outdim)))
        rnn = DeterministicSensoryRosslerPolicy(spec,
                                 env_timestep=step,
                                 timeconstant=timeconstant,
                                 integrator='euler')
        rnn.get_param_values()
        
    def testCoupling(self):
        indim, outdim = 1,2
        step = self.TIMESTEP
        timeconstant = self.TIMECONSTANT
        spec = SyncEnvSpec(DummySpace(indim),
                           DummySpace(outdim),
                           np.zeros((outdim, outdim)))
        rnn = DeterministicRosslerPolicy(spec,
                                         hidden_sizes=(outdim,),
                                         env_timestep=step,
                                         timeconstant=timeconstant,
                                         integrator='rk4',
                                         state_initf=init_fixedoffset)
        # no coupling         
        x = np.array([1])
        orbit = []
        nsteps = int(timeconstant*100/step)
        for _ in range(nsteps):
            _ = rnn.output(x, step)
            orbit.append( rnn.get_state()['h'] )
           
        # turn on coupling
        K = np.zeros((outdim, outdim))
        K[range(0,outdim-1), range(1,outdim)] = 2*np.pi
        rnn._layers[0].set_K(K)
           
        for _ in range(nsteps):
            _ = rnn.output(x, step)
            orbit.append( rnn.get_state()['h'] )
        orbit = np.array(orbit)
           
        plt.figure()
        plt.plot(orbit[:,0])
        plt.plot(orbit[:,3])
        plt.plot([nsteps,nsteps],[-5,5], 'k')
        plt.title('Coupling (onset at t=1000)')       
    
    def testThreeDimCoupling(self):
        indim, outdim = 1,3
        step = self.TIMESTEP
        timeconstant = self.TIMECONSTANT
        spec = SyncEnvSpec(DummySpace(indim),
                           DummySpace(outdim),
                           np.zeros((outdim, outdim)))
        rnn = DeterministicRosslerPolicy(spec,
                                         hidden_sizes=(outdim,),
                                         env_timestep=step,
                                         timeconstant=timeconstant,
                                         integrator='rk4',
                                         state_initf=init_fixedoffset)
        # no coupling         
        x = np.array([1])
        orbit = []
        nsteps = int(timeconstant*100/step)
        for _ in range(nsteps):
            _ = rnn.output(x, step)
            orbit.append( rnn.get_state()['h'] )
           
        # turn on coupling
        K = np.zeros((outdim, outdim))
        K[0,1] = np.pi      # node 1 in antiphase with node 0
        K[0,2] = 2*np.pi    # node 2 attempts in phase with node 0
        K[1,2] = 2*np.pi    # node 2 attempts in phase with node 1 --> mean field of node 0 & 1
        rnn._layers[0].set_K(K)
           
        for _ in range(nsteps):
            _ = rnn.output(x, step)
            orbit.append( rnn.get_state()['h'] )
        orbit = np.array(orbit)
           
        plt.figure()
        plt.plot(orbit[:,0])
        plt.plot(orbit[:,3])
        plt.plot(orbit[:,6])
        plt.plot([nsteps,nsteps],[-5,5], 'k')
        plt.title('3D Coupling (onset at t=1000)')
        plt.show()
        
    def testEuler(self):
        indim, outdim = 1,1
        step = self.TIMESTEP
        timeconstant = self.TIMECONSTANT
        spec = SyncEnvSpec(DummySpace(indim),
                           DummySpace(outdim),
                           np.zeros((outdim, outdim)))
        rnn = DeterministicRosslerPolicy(spec,
                                         hidden_sizes=(outdim,),
                                         env_timestep=step,
                                         timeconstant=timeconstant,
                                         integrator='euler')
         
        rnn._layers[-1].W = np.eye(rnn._layers[-1].W.shape[0])
        rnn._layers[-1].b = np.zeros(rnn._layers[-1].b.shape)
            
        x = np.array([1])
        orbit = []
        for _ in range(int(timeconstant*100/step)):
            _ = rnn.output(x, step)
            orbit.append( rnn.get_state()['h'] )
        orbit = np.array(orbit)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
        plt.title('Euler')
  
    def testRK4(self):
        indim, outdim = 1,1
        step = self.TIMESTEP
        timeconstant = self.TIMECONSTANT
        spec = SyncEnvSpec(DummySpace(indim),
                           DummySpace(outdim),
                           np.zeros((outdim, outdim)))
        rnn = DeterministicRosslerPolicy(spec,
                                         hidden_sizes=(outdim,),
                                         env_timestep=step,
                                         timeconstant=timeconstant,
                                         integrator='rk4')
                   
        rnn._layers[-1].W = np.eye(rnn._layers[-1].W.shape[0])
        rnn._layers[-1].b = np.zeros(rnn._layers[-1].b.shape)
           
        x = np.array([1])
        orbit = []
        for _ in range(int(timeconstant*100/step)):
            _ = rnn.output(x, step)
            orbit.append( rnn.get_state()['h'] )
        orbit = np.array(orbit)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
        plt.title('RK4')
        plt.show()
              
#     def testCoupling(self):
#         indim, outdim = 1,2
#         step = self.TIMESTEP
#         timeconstant = self.TIMECONSTANT
#         rnn = DeterministicRosslerPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)),
#                                          hidden_sizes=(outdim,),
#                                          env_timestep=step,
#                                          timeconstant=timeconstant,
#                                          integrator='rk4',
#                                          state_initf=init_fixedoffset)
#            
#         rnn._layers[0].k = 0.0 # turn off coupling
#         rnn._layers[0].phi = 0.0
#            
#         x = np.array([1])
#         orbit = []
#         nsteps = int(timeconstant*100/step)
#         for _ in range(nsteps):
#             _ = rnn.output(x, step)
#             orbit.append( rnn.get_state()['h'] )
#            
#         # turn on coupling
#         rnn._layers[0]._K[range(0,outdim-1), range(1,outdim)] = 1.0 
#         rnn._layers[0].k = 0.5
#         rnn._layers[0].phi = 0.5*np.pi
#          
#         for _ in range(nsteps):
#             _ = rnn.output(x, step)
#             orbit.append( rnn.get_state()['h'] )
#         orbit = np.array(orbit)
#            
#         plt.figure()
#         plt.plot(orbit[:,0])
#         plt.plot(orbit[:,3])
#         plt.plot([nsteps,nsteps],[-5,5], 'k')
#         plt.title('Coupling (onset at t=1000)')
#         plt.show()
        
#     def testEuler(self):
#         indim, outdim = 1,1
#         step = self.TIMESTEP
#         timeconstant = self.TIMECONSTANT
#         rnn = DeterministicSensoryRosslerPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)),
#                                          hidden_sizes=(outdim,),
#                                          env_timestep=step,
#                                          timeconstant=timeconstant,
#                                          integrator='euler')
#         rnn._layers[0]._K = np.zeros(rnn._layers[0]._K.shape)
#         rnn._layers[-1].W = np.eye(rnn._layers[-1].W.shape[0])
#         rnn._layers[-1].b = np.zeros(rnn._layers[-1].b.shape)
#           
#         x = np.array([1])
#         orbit = []
#         for _ in range(int(timeconstant*100/step)):
#             _ = rnn.output(x, step)
#             orbit.append( rnn.get_state()['h'] )
#         orbit = np.array(orbit)
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
#         plt.title('Euler')
#           
#           
#     def testRK4(self):
#         indim, outdim = 1,1
#         step = self.TIMESTEP
#         timeconstant = self.TIMECONSTANT
#         rnn = DeterministicRosslerPolicy(EnvSpec(DummySpace(indim), DummySpace(outdim)),
#                                          hidden_sizes=(outdim,),
#                                          env_timestep=step,
#                                          timeconstant=timeconstant,
#                                          integrator='rk4')
#                   
#         rnn._layers[0]._K = np.zeros(rnn._layers[0]._K.shape)
#         rnn._layers[-1].W = np.eye(rnn._layers[-1].W.shape[0])
#         rnn._layers[-1].b = np.zeros(rnn._layers[-1].b.shape)
#           
#         x = np.array([1])
#         orbit = []
#         for _ in range(int(timeconstant*100/step)):
#             _ = rnn.output(x, step)
#             orbit.append( rnn.get_state()['h'] )
#         orbit = np.array(orbit)
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
#         plt.title('RK4')
#         plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()