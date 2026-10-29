import unittest
import numpy as np
from p006_labs import energy,run
from p007_thomson import energy_and_gradient,unit,relax,defect_indices

class P006Tests(unittest.TestCase):
    def test_integer_energy_and_symmetry(self):
        self.assertEqual(energy(np.array([1,1,1])),5)
        a=np.random.default_rng(3).choice([-1,1],40)
        self.assertEqual(energy(a),energy(a[::-1]))
        self.assertEqual(energy(a),energy(-a))
        self.assertEqual(energy(a),energy(a*np.array([(-1)**i for i in range(40)])))
    def test_repeatable(self):
        self.assertEqual(run(20,50,3,'neural'),run(20,50,3,'neural'))
    def test_official_baseline(self):
        s='++++++--+++++-----++-++--+--++++++++++---+-++--++++-+--++----+--+-+-+-+-++---++---++-+-++-+-++--+-+-'
        self.assertEqual(len(s),100)
        self.assertEqual(energy(np.array([1 if c=='+' else -1 for c in s])),578)

class P007Tests(unittest.TestCase):
    def test_gradient_and_relaxation(self):
        x=unit(np.random.default_rng(1).normal(size=(7,3)))
        _,g=energy_and_gradient(x.ravel())
        xp=x.ravel().copy();xm=x.ravel().copy();h=1e-6
        xp[3]+=h;xm[3]-=h
        fd=(energy_and_gradient(xp)[0]-energy_and_gradient(xm)[0])/(2*h)
        self.assertAlmostEqual(fd,g[3],places=4)
        y,E,_=relax(x,40)
        self.assertLessEqual(E,energy_and_gradient(x.ravel())[0]+1e-8)
        self.assertLess(np.max(abs(np.linalg.norm(y,axis=1)-1)),1e-12)
        self.assertTrue(len(defect_indices(y))>0)
if __name__=='__main__':unittest.main()
