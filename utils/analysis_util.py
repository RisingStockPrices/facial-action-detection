import shutil
import numpy as np
import os
from sklearn.cluster import KMeans


"""
Cluster by AU score
"""
def cluster_faces(aus,debug=False,n_clusters=20,ROOT_DIR='./data/'):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(aus)
    # load each column as np.array
    res = kmeans.predict(aus)
    bins = {}
    import pdb;pdb.set_trace()
    for i,re in enumerate(res):
        if re in bins.keys():
            bins[re].append(i)
        else:
            bins[re] = [i]
    # save
    if debug:
        src_dir = os.path.join(ROOT_DIR,'frames','wink_test')
        for idx,lst in bins.items():
            dir = os.path.join(ROOT_DIR,'tmp','bin_%d'%idx)
            if not os.path.exists(dir):
                os.makedirs(dir)
            for f in lst:
                fname = os.path.join(src_dir,'%d.png'%f)
                dst_pth = os.path.join(dir,'%d.png'%f)
                shutil.copyfile(fname,dst_pth)

    return bins

"""
Sort by activation score (deprecated)
This is probably a bad idea...
"""
def sort_au(aus):
    # print list of activated aus
    sorted_aus = []
    for au in aus:
        res = np.argsort(au)
        # import pdb;pdb.set_trace()
        sorted_aus.append(res)
    print(sorted_aus)


def sort_by_similarity(aus, target, loss='L2',top=None,threshold=None):
    # df to np array
    aus = aus.to_numpy()[:-1]
    target = aus[target]
    
    if loss=='L2':
        sim_scores = np.linalg.norm(aus-target,axis=1) #mean_squared_error(target,scores,multioutput='rawvalues')
        idx_sorted = np.argsort(sim_scores)
    else:
        raise NotImplemented

    if top is not None:
        mode='top'
    elif threshold is not None:
        mode='threshold'

    # sort by similarity score
    frames_sorted = [(idx,sim_scores[idx]) for idx in idx_sorted]

    # return according to mode (topk or threshold)
    if mode is 'top':
        return frames_sorted[:top]
    else:
        for idx,(_,score) in enumerate(frames_sorted):
            if score > threshold:
                break
        res = frames_sorted[:idx]
        return res
        res = (np.array([i for _,i in frames_sorted])>threshold) * frames_sorted
        return res
