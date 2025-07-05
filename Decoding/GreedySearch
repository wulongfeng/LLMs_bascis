import torch
import torch.nn.functional as F

def greedy_search(logits, max_len=20, eos_token_id=2):
    batch_size = logits.size(0)
    seq_len = logits.size(1)
    vocab_size = logits.size(2)

    generated_sequence = torch.zeros(batch_size, max_len, dtype=torch.long)

    current_input = torch.zeros(batch_size, dtype=torch.long) # initial token id

    for t in range(max_len):
        currrent_logits = logits[:, t, :]
        probs = F.softmax(currrent_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)

        generated_sequence[:, t] = next_token

        #if torch.all(next_token == eos_token_id):
        if (next_token == eos_token_id).all():
            break

        current_input = next_token
    
    return generated_sequence

