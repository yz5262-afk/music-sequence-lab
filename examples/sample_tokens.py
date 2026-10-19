from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sampling import sample_next_token
torch.manual_seed(7)
logits=torch.tensor([0.1,1.2,2.0,1.4,0.3])
history=[]
for _ in range(40):
    history.append(sample_next_token(logits,temperature=.9,top_k=3,recent_tokens=history))
print('Synthetic token IDs (not trained music):',history)
