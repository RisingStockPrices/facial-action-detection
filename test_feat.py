import os
import shutil
import pandas as pd
import numpy as np

from utils.util import save_subplots
from feat import Detector
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

ROOT_DIR = './data/'
FRAME_DIR = os.path.join(ROOT_DIR,'frames')
SRC_DIR = os.path.join(ROOT_DIR,'faces')
AUS_DIR = os.path.join(ROOT_DIR,'aus')
detector = None  

def load_model():
    face_model = "retinaface"
    landmark_model = "mobilenet"
    au_model = "rf"
    emotion_model = "resmasknet"
    detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)
    return detector


def extract_aus(frames,name):
    image_prediction = detector.detect_image(frames)
    res = image_prediction.aus()
    import pdb;pdb.set_trace()
    pth = os.path.join(ROOT_DIR,'aus','%s.csv'%name)
    res.to_csv(pth)
    return res


def sort_by_similarity(aus, target, loss='L2',top=5):
    # df to np array
    aus = aus.to_numpy()[:-1]
    target = aus[target]

    if loss=='L2':
        sim_scores = np.linalg.norm(aus-target,axis=1) #mean_squared_error(target,scores,multioutput='rawvalues')
        idx_sorted = np.argsort(sim_scores)
    else:
        raise NotImplemented

    # sort by similarity score
    frames_sorted = [(idx,sim_scores[idx]) for idx in idx_sorted]
    return frames_sorted[:top]

"""
Cluster by AU score
"""
def cluster_faces(aus,debug=False,n_clusters=20):
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

"""
Extract AU scores directly from video
Not from face frames
"""
def extract_aus_from_video(pth,save_aus=True):
    # test_video = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")
    name = pth.split('/')[-1].split('.')[0]
    aus_pth = os.path.join(AUS_DIR,'%s.csv'%name)
    # already exists, simply load
    if os.path.exists(aus_pth):
        res = pd.read_csv(aus_pth)
        return res

    detector = load_model()
    video_prediction = detector.detect_video(pth)
    res = video_prediction.aus()
    if save_aus:
        import pdb;pdb.set_trace()
        res.to_csv(aus_pth,index=False)
        res.plot().figure.savefig(aus_pth.replace('csv','png'))
    return res

def plot_aus(aus,pth='aus_tmp.png'):
    res = aus.plot()
    res.figure.savefig(pth)

""" 
Detect single frame action
"""
def detect_single_frame(vid_pth,target=63,top_k=20):
    aus = extract_aus_from_video(vid_pth)
    res = sort_by_similarity(aus,target,top=top_k)
    print('='*50)
    print('Detecting Single Frame action! Target Frame: %d'%target)
    print('='*50)
    print('Top %d matches and scores' % top_k)
    
    for frame,score in res:
        print(frame,score)
    return res


if __name__ == "__main__":
    
    frame_dir = os.path.join(FRAME_DIR,'wink_test')
    frames = [os.path.join(frame_dir,i) for i in os.listdir(frame_dir)] #EST_SRC_DIR) ["0.png","1.png","10.png","100.png","101.png","102.png","103.png"]
    

    vid_pth = os.path.join(SRC_DIR,'wink_test.avi')
    detect_single_frame(vid_pth)
    


