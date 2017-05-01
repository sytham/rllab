import unittest
import numpy as np

from contrib.rnn.policies.base import Layer, RecurrentLayer, identity, logisticmap

class TestLayer(unittest.TestCase):

    def testInit(self):
        indim, outdim = 3,2
        self.assertRaises(ValueError, Layer, 0, 1)
        self.assertRaises(ValueError, Layer, 1, 0)
        
        # no bias
        l = Layer(indim,outdim,transferf=identity,bias=False)
        self.assertTrue( l.encode().shape[0] == indim*outdim)
        
        W = np.array([[1,2,3],[4,5,6]])
        l.decode(W.ravel())
        self.assertTrue( np.all(l.W == W))
        self.assertTrue( np.all(l.b == np.zeros((outdim,))))
        self.assertTrue( np.all(l.encode() == W.ravel()))
        
        # bias
        l = Layer(indim,outdim,transferf=identity,bias=True)
        self.assertTrue( l.encode().shape[0] == indim*outdim + outdim)
        
        W = np.array([[1,2,3],[4,5,6]])
        self.assertRaises(AssertionError, l.decode, W.ravel())
        b = np.array([1,1])
        l.decode(np.concatenate([W.ravel(), b]))
        
        self.assertTrue( np.all(l.W == W))
        self.assertTrue( np.all(l.b == b))
        self.assertTrue( np.all(l.encode() == np.concatenate([W.ravel(), b])))
        
    def testIdentity(self):
        indim, outdim = 3,2
        l = Layer(indim,outdim,transferf=identity,bias=False)
        l.decode( np.array([[1,2,3],[4,5,6]]).ravel() )
        x = np.array([1,2,3])
        self.assertTrue(np.all(l.out(x) == np.array([14,32])))
    
        l = Layer(indim,outdim,transferf=identity,bias=True)
        l.decode( np.concatenate([np.array([[1,2,3],[4,5,6]]).ravel(), np.array([1,2])]) )
        self.assertTrue(np.all(l.out(x) == np.array([15,34])))
    
    def testTanh(self):
        indim, outdim = 3,2
        l = Layer(indim,outdim,bias=False)
        l.decode( np.array([[1,2,3],[4,5,6]]).ravel() )
        x = np.array([1,2,3])
        self.assertTrue(np.all(l.out(x) == np.tanh(np.array([14,32]))))
    
        l = Layer(indim,outdim,bias=True)
        l.decode( np.concatenate([np.array([[1,2,3],[4,5,6]]).ravel(), np.array([1,2])]) )
        self.assertTrue(np.all(l.out(x) == np.tanh(np.array([15,34]))))
    
    def testLogmap(self):
        indim, outdim = 3,2
        l = Layer(indim,outdim,transferf=logisticmap, bias=False)
        l.decode( np.array([[1,2,3],[4,5,6]]).ravel() )
        x = np.array([1,2,3])
        self.assertTrue(np.all(l.out(x) == logisticmap(np.array([14,32]))))
    
        l = Layer(indim,outdim,transferf=logisticmap, bias=True)
        l.decode( np.concatenate([np.array([[1,2,3],[4,5,6]]).ravel(), np.array([1,2])]) )
        self.assertTrue(np.all(l.out(x) == logisticmap(np.array([15,34]))))

class TestRecurrentLayer(unittest.TestCase):

    def testInit(self):
        indim, outdim = 3,2
        self.assertRaises(ValueError, RecurrentLayer, 0, 1)
        self.assertRaises(ValueError, RecurrentLayer, 1, 0)
        
        # no bias
        l = RecurrentLayer(indim,outdim,transferf=identity,bias=False)
        self.assertTrue( l.encode().shape[0] == indim*outdim + outdim**2)
        
        W = np.array([[1,2,3],[4,5,6]])
        K = np.array([[10,11],[12,13]])
        self.assertRaises(AssertionError, l.decode, W.ravel())
        theta = np.concatenate([W.ravel(), K.ravel()])
        l.decode(theta)
        self.assertTrue( np.all(l.W == W))
        self.assertTrue( np.all(l.b == 0))
        self.assertTrue( np.all(l.K == K))
        self.assertTrue( np.all(l.encode() == theta))
        
        # bias
        l = RecurrentLayer(indim,outdim,transferf=identity,bias=True)
        self.assertTrue( l.encode().shape[0] == indim*outdim + outdim + outdim**2)
        self.assertRaises(AssertionError, l.decode, theta)
        b = np.array([1,2])
        theta = np.concatenate([W.ravel(), b, K.ravel()])
        l.decode(theta)
        self.assertTrue( np.all(l.W == W))
        self.assertTrue( np.all(l.b == b))
        self.assertTrue( np.all(l.K == K))
        self.assertTrue( np.all(l.encode() == theta))
        
    def testIdentity(self):
        indim, outdim = 3,2
        l = RecurrentLayer(indim,outdim,transferf=identity,bias=False)
        l.decode( np.concatenate([np.array([[1,2,3],[4,5,6]]).ravel(), np.zeros((outdim**2,))]) )
        x = np.array([1,2,3])
        self.assertTrue(np.all(l.out(x) == np.array([14,32])))
    
        l = RecurrentLayer(indim,outdim,transferf=identity,bias=True)
        l.decode( np.concatenate([np.array([[1,2,3],[4,5,6]]).ravel(), np.array([1,2]), np.zeros((outdim**2,))]) )
        self.assertTrue(np.all(l.out(x) == np.array([15,34])))
    
        # add recurrent connections
        l.decode( np.concatenate([np.array([[1,2,3],[4,5,6]]).ravel(), np.array([1,2]), np.eye(outdim).ravel()]) )
        x = np.ones((indim,))
        self.assertTrue(np.all(l.out(x) == np.array([22,51])))
          
        # add lateral connections
        l.decode( np.concatenate([np.array([[1,2,3],[4,5,6]]).ravel(), np.array([1,2]), np.array([[0.9, 0.1], [0.1, 0.9]]).ravel()]) )
        o = l.out(x)
        self.assertTrue(np.all(o == np.array([0.9*22 + 0.1*51 + 7, 0.9*51 + 0.1*22 + 17])))

            


if __name__ == "__main__":
    unittest.main()