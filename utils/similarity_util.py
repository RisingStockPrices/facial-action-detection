import numpy as np

def find_matches(seq,target_seq,mode='sliding-window',loss='l2',top=None,threshold=None):
    # sliding window
    target_len = target_seq.shape[0]
    # import pdb;pdb.set_trace()
    alignment_scores = []
    for i in range(len(seq)-target_len):    
        score = np.linalg.norm(seq[i:i+target_len]-target_seq)
        score /= target_len
        alignment_scores.append(score)

    idx_sorted = np.argsort(alignment_scores)
    windows_sorted = [(idx,idx+target_len,alignment_scores[idx]) for idx in idx_sorted]


    if top != None:
        return windows_sorted[:top]
    elif threshold != None:
        for idx,(_,_,score) in enumerate(windows_sorted):
            if score > threshold:
                break
        return windows_sorted[:idx]
    else:
        raise NotImplemented

def prune_matches(matches,seq_len,margin=2):
    occupancy = np.zeros(seq_len)

    pruned_matches = []
    for start,end,score in matches:

        if (occupancy[start:end]==False).all():
            pruned_matches.append((start,end,score))
            occupancy[start:end] = True
        # print(occupancy)
    
    return pruned_matches