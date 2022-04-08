import os
import pandas as pd
import numpy as np

from utils.util import save_subplots
from utils.analysis_util import sort_by_similarity
from utils.plot_util import color_matches,test_face_plotter
from utils.similarity_util import find_matches, prune_matches
from feat import Detector

ROOT_DIR = './data/'
FRAME_DIR = os.path.join(ROOT_DIR,'frames')
SRC_DIR = os.path.join(ROOT_DIR,'faces')
AUS_DIR = os.path.join(ROOT_DIR,'aus')
AUSPLOT_DIR = os.path.join(ROOT_DIR,'aus_plot')

detector = None

def load_model():
    face_model = "retinaface"
    landmark_model = "mobilenet"
    au_model = "rf"
    emotion_model = "resmasknet"
    detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)
    return detector


def extract_aus_from_frames(frames,name):
    image_prediction = detector.detect_image(frames)
    res = image_prediction.aus()
    import pdb;pdb.set_trace()
    pth = os.path.join(ROOT_DIR,'aus','%s.csv'%name)
    res.to_csv(pth)
    return res


"""
Extract AU scores directly from video
Not from face frames
"""
def extract_aus_from_video(pth,save_aus=True):
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


""" 
Detect single frame action
"""
def detect_single_frame(vid_pth,target=63,top_k=20,return_aus=False,threshold=None):
    aus = extract_aus_from_video(vid_pth)
    res = sort_by_similarity(aus,target,threshold=threshold)
    print('='*50)
    print('Detecting Single Frame action! Target Frame: %d'%target)
    print('='*50)
    
    for frame,score in res:
        print(frame,score)
    if return_aus:
        return aus,res
    else:
        return res

def detect_multi_frames(vid_pth,target=(1,5),top_k=None,threshold=None,return_aus=False):
    aus_ = extract_aus_from_video(vid_pth)
    aus = aus_.to_numpy()[:-1]

    print('='*50)
    print('Detecting Multi-Frame action! Target Frame: %d~%d'%(target[0],target[1]))
    print('='*50)

    target_seq = aus[target[0]:target[1]]
    matches = find_matches(aus,target_seq,top=top_k,threshold=threshold)
    pruned = prune_matches(matches,len(aus))
    print(pruned)
    
    if return_aus:
        return aus_,pruned
    else:
        return pruned

"""
Generalized version of detect multi frames
when target action is not from video to search
"""
def detect_facial_expression(source,target,threshold=None,top_k=None):
    matches = find_matches(source,target,top=top_k,threshold=threshold)
    pruned = prune_matches(matches,len(source))
    # print(pruned)
    return pruned

if __name__ == "__main__":
    
    # frame_dir = os.path.join(FRAME_DIR,'wink_test')
    # frames = [os.path.join(frame_dir,i) for i in os.listdir(frame_dir)] #EST_SRC_DIR) ["0.png","1.png","10.png","100.png","101.png","102.png","103.png"]

    vid_pth = os.path.join(SRC_DIR,'test2.mp4')
    target_pth = os.path.join(SRC_DIR,'wink_test.avi')

    threshold = 0.15
    aus_input_ = extract_aus_from_video(vid_pth)[:-1]
    aus_input = aus_input_.to_numpy()
    aus_tar = extract_aus_from_video(target_pth)[:-1].to_numpy()
    
    
    test_face_plotter(aus_input)

    import pdb;pdb.set_trace()
    target_idx = (34,44)
    target = aus_tar[target_idx[0]:target_idx[1]]
    res = detect_facial_expression(aus_input,target,threshold=threshold)
    # aus,res = detect_single_frame(vid_pth,return_aus=True,threshold=threshold)
    # aus,res = detect_multi_frames(vid_pth,target=(33,44),threshold=threshold, return_aus=True)
    
    pth = os.path.join(AUSPLOT_DIR,'diff_source_threshold_%.2f.png'%threshold)
    color_matches(aus_input_,res,pth=pth,title='threshold is %f'%threshold)