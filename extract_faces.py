import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
# import model
import os
import argparse

"""
Extract & Crop 1 main face in a video
using MediaPipe library
"""


#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('vid_pth', type=str,help='Path of the video to extract faces from')
parser.add_argument('--out_dir', default='./data',type=str,help='Dir to store results')
parser.add_argument('--debug', default=False,type=bool)
args = parser.parse_args()


FACE_WIDTH = 400
FACE_HEIGHT = 400
FRAME_RATE = 24
MAX_FRAMES = None if not args.debug else 10

def extract_face(image,bbox):

  # (h,w,c) PIL image
  height,width,_ = image.shape

  xmin = int(bbox.xmin * width)
  ymin = int(bbox.ymin * height)
  bbox_w = int(bbox.width*width)
  bbox_h = int(bbox.height*height)
  #     "xmax" : int(bbox.width * width + bbox.xmin * width),
  #     "ymax" : int(bbox.height * height + bbox.ymin * height)
  # }

  # if bbox_w ==0 or bbox_h ==0 or xmin+bbox_w>width:
    # import pdb;pdb.set_trace()
  # model.tensor_to_image(image)
  img = image[max(0,ymin):min(height,ymin+bbox_h),max(0,xmin):min(width,xmin+bbox_w),:]
  # model.tensor_to_image(img)


  return img



if __name__=="__main__":
    
    vid_pth = args.vid_pth #'/home/spock-the-wizard/cmu/sg/emonet/test/data/wink_test.mp4'
    vid_name = vid_pth.split('/')[-1].split('.')[0]
    vid_out_pth = os.path.join(args.out_dir,'faces','%s.avi'%vid_name)
    frame_dir = os.path.join(args.out_dir,'frames',vid_name)
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    cap = cv2.VideoCapture(vid_pth)
    out = cv2.VideoWriter(vid_out_pth,cv2.VideoWriter_fourcc('M','J','P','G'),FRAME_RATE, (FACE_WIDTH,FACE_HEIGHT))

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_detection = mp.solutions.face_detection #mesh = mp.solutions.face_mesh
    
    cnt = 0
    with mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            if MAX_FRAMES and cnt > MAX_FRAMES:
                break

            success,image = cap.read()
            if not success:
                break

            results = face_detection.process(image)
            
            # image = cv2.cvtColor(image)
      
            if results.detections:
                for detection in results.detections:

                    face = extract_face(image,detection.location_data.relative_bounding_box)
                    face = cv2.resize(face,(FACE_WIDTH,FACE_HEIGHT))

                    # write file to video
                    out.write(face)
                    # save as separate file
                    pth = os.path.join(frame_dir,'%d.png'%cnt)
                    cv2.imwrite(pth,face)
            cnt += 1

    cap.release()
    out.release()