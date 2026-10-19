import unittest
import torch
from sampling import sample_next_token
class SamplingTests(unittest.TestCase):
    def test_repeated_negative_logit_is_penalized(self):
        # The previous sign-agnostic division would favor token 0 here.
        x=torch.tensor([-1.,-1.1])
        for _ in range(10):
            self.assertEqual(sample_next_token(x,top_k=1,recent_tokens=[0],repetition_penalty=2),1)
        self.assertEqual(x.tolist(),[-1.,x[1].item()])
    def test_top_k_never_leaves_candidate_set(self):
        torch.manual_seed(7)
        for _ in range(50):
            self.assertIn(sample_next_token(torch.tensor([0.,2.,3.,-1.]),top_k=2),[1,2])
if __name__=='__main__': unittest.main()
