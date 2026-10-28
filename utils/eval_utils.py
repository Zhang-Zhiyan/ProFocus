import torch
import torch.nn.functional as F

def top_filtering1(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
   
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
       
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probabilities > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() 
        sorted_indices_to_remove[..., 0] = 0
    
        bt = logits.size(0)
        for i in range(bt):
            indices_to_remove =  sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

