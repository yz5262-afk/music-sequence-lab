"""Token sampler exported from the music notebook; independent of the MIDI corpus."""
import torch

def sample_next_token(logits, temperature=1.1, top_k=20,
                      recent_tokens=None, repetition_penalty=1.2):

    logits = logits.clone()

    # 🔥 Repetition penalty
    if recent_tokens is not None:
        for token in set(recent_tokens[-20:]):
            logits[token] = torch.where(logits[token] < 0, logits[token] * repetition_penalty, logits[token] / repetition_penalty)

    # Temperature
    logits = logits / max(temperature, 1e-6)

    # Top-k
    top_k = min(top_k, logits.shape[-1])
    top_logits, top_indices = torch.topk(logits, k=top_k)

    probs = torch.softmax(top_logits, dim=-1)

    idx = torch.multinomial(probs, 1).item()
    return top_indices[idx].item()
