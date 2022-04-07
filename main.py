import os
import pandas as pd
import numpy as np

from utils.util import save_subplots
from utils.analysis_util import sort_by_similarity

from feat import Detector

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
