import os
import pandas as pd
import numpy as np
import argparse

from utils.util import save_subplots
from utils.analysis_util import sort_by_similarity
from utils.plot_util import color_matches, save_detection_results,test_face_plotter
from utils.similarity_util import find_matches, prune_matches

from library import *

parser = argparse.ArgumentParser()
parser.add_argument('--video',type=str,help='name of video file')
parser.add_argument('--action',type=str,help='name of facial action')
parser.add_argument('--threshold',default=0.15,type=float,help='detection threshold')
args = parser.parse_args()

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

    video_pth = os.path.join(SRC_DIR,args.video)
    facial_action = args.action #'l_wink'
    threshold = args.threshold

    aus_input_ = extract_aus_from_video(video_pth)[:-1]
    aus_input = aus_input_.to_numpy()

    # extract target from library
    target = FAL.retrieve_aus(facial_action)

    res = detect_facial_expression(aus_input,target,threshold=threshold)
    
    
    save_detection_results(aus_input_,res,args.video,facial_action,threshold)
    # pth = os.path.join(AUSPLOT_DIR,'%s_diff_source_threshold_%.2f.png'%threshold)
    # color_matches(aus_input_,res,pth=pth,title='athreshold is %f'%threshold)