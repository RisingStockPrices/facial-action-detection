from feat import Detector

import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video',default='./data/faces/face1.mp4',type=str,help='path of video to extract frames')

args = parser.parse_args()

ROOT_DIR = './data/'
FRAME_DIR = os.path.join(ROOT_DIR,'frames')
SRC_DIR = os.path.join(ROOT_DIR,'faces')
AUS_DIR = os.path.join(ROOT_DIR,'aus')
AUSPLOT_DIR = os.path.join(ROOT_DIR,'aus_plot')


def load_model():
    face_model = "retinaface"
    landmark_model = "mobilenet"
    au_model = "rf"
    emotion_model = "resmasknet"
    detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)
    return detector

"""TODO: fix this"""
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
        # import pdb;pdb.set_trace()
        res.to_csv(aus_pth,index=False)
        res.plot().figure.savefig(aus_pth.replace('csv','png'))
    return res


if __name__ == "__main__":
    extract_aus_from_video(args.video)