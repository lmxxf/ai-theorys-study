import unittest
import torch
from cold_objective import cold_loss, cold_recall

class ColdObjectiveTests(unittest.TestCase):
    def test_excluded_candidates_and_labels(self):
        z=torch.tensor([[1.,2.,3.,4.],[0.,1.,2.,3.]],requires_grad=True)
        y=torch.tensor([[0,2],[1,3]])
        cached=torch.tensor([[True,False,False,False],[True,True,True,True]])
        loss=cold_loss(z,y,cached)
        expected=torch.logsumexp(torch.tensor([2.,3.,4.]),0)-3
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue((z.grad[cached]==0).all())
        self.assertTrue((z.grad[1]==0).all())
        modified=z.detach().clone();modified[cached]=10000
        torch.testing.assert_close(cold_loss(modified,y,cached),loss)
        self.assertEqual(cold_recall(z,y,cached,k=1),0)
        self.assertEqual(cold_recall(z,y,cached,k=2),1)

    def test_all_cached_is_zero_with_zero_gradient(self):
        z=torch.randn(3,4,requires_grad=True);y=torch.tensor([[0,1]]*3)
        loss=cold_loss(z,y,torch.ones_like(z,dtype=torch.bool));loss.backward()
        self.assertEqual(loss.item(),0)
        self.assertTrue((z.grad==0).all())

if __name__=='__main__':unittest.main()
