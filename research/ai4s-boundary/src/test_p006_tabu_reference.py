import unittest
import numpy as np
from p006_tabu_reference import correlations,all_flip_energies,one_flip,tabu_search
class TestTabu(unittest.TestCase):
    def test_exact_delta_random(self):
        rng=np.random.default_rng(8)
        for n in [8,16,40,100]:
            for _ in range(5):
                s=rng.choice([-1,1],size=n).astype(np.int64)
                c=correlations(s);flipE=all_flip_energies(s,c)
                for j in range(n):
                    t=s.copy();t[j]*=-1
                    co=correlations(t)
                    self.assertTrue(np.array_equal(one_flip(s,c,j),co))
                    self.assertEqual(int(co@co),int(flipE[j]))
    def test_tabu_repeatable(self):
        a=tabu_search(30,9,100);b=tabu_search(30,9,100)
        for key in ['initial','best_energy','best_sequence','moves_evaluated']:
            self.assertEqual(a[key],b[key])
if __name__=='__main__':unittest.main()
