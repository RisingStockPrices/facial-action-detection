import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import torch
import os
import argparse
from model import *


from sklearn.metrics.pairwise import cosine_similarity

"""
Code to test the feature extractor in emonet
"""


def extract_features(frame_dir,outpath):
    feats_list = []
    for fname in os.listdir(frame_dir):
        img = cv2.imread(os.path.join(frame_dir,fname))
        
        feats = get_aroval(img,return_feat=True)
        feats_list.append(feats.cpu().numpy())
        # save as numpy file
    res = np.array(feats_list)#feats_list.cpu().numpy()
    np.save(outpath,res)
    return res
    # name = frame_dir.split('/')[-1]#fname.split('.')[0]
    # pth = os.path.join(feat_dir,name+'.npy')
    # np.save(pth,feats.cpu().numpy())

def save_scores(scores,outpath):
    # create file

    # write all the scores

    raise NotImplemented

def save_top_k_images(frame_dir,res,outpth,num_img=5):
    # create a grid
    res = res[:num_img]
    out = []
    for idx,score in res:
        # read image
        pth_img = os.path.join(frame_dir,'%d.png'%idx)
        img = cv2.imread(pth_img)
        zeros = np.zeros((500,400,3))
        zeros[:400,:,:] = img
        # img = cv2.resize(img,(400,500))
        img_ = cv2.putText(zeros, str(score), (10,430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2, cv2.LINE_AA)
        out.append(img_)
        # wanna check if size consistent
    out = np.array(out)
    out = np.concatenate(out,axis=1)
    cv2.imwrite(outpth,out)


if __name__ == "__main__":

    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_dir', type=str,help='Path of frame dir of extracted faces')
    parser.add_argument('--out_dir', default='./data',type=str,help='Dir to store results')
    parser.add_argument('--TARGET_FRAME', default=1, type=int)
    args = parser.parse_args()
    
    name = args.frame_dir.split('/')[-1]
    pth_feats = os.path.join(args.out_dir,'feats',name+'.npy')
    if not os.path.exists(pth_feats):
        feats = extract_features(args.frame_dir,pth_feats)
    else:
        feats = np.load(pth_feats)
    
    target_list = [0,20,30,50,70,99,100,34,200,250,178,209]
    for target_frame in target_list:
        print('working on frame %d'%target_frame)
        feats = feats.squeeze() # N_FRAMES X DIM
        target = feats[target_frame].reshape(1,-1)
        
        # import pdb;pdb.set_trace()
        scores = cosine_similarity(feats,target).squeeze()

        # visualize score for each frame as text
        out_dir = os.path.join(args.out_dir,'cosine_sim',name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        pth_scores = os.path.join(out_dir,'scores.txt')
        # save_scores(scores,pth_scores)

        # gather highest scores
        sorted_idx = np.argsort(-scores)
        res = [(i,scores[i]) for i in sorted_idx]
        pth_res = os.path.join(out_dir,'target_%d.png'%target_frame)
        save_top_k_images(args.frame_dir,res,pth_res)

        res.reverse()
        pth_res = os.path.join(out_dir,'target_%d_rev.png'%target_frame)
        save_top_k_images(args.frame_dir,res,pth_res)
        # visualize top K scores in a grid and save
    
