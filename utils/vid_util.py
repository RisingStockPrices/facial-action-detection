import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video',default='./data/faces/face1.mp4',type=str,help='path of video to extract frames')

args = parser.parse_args()

FRAME_DIR = './data/frames/'
PRINT_STATUS_FREQUENCY=100
def extract_frames(vid_pth,frame_dir=None):
    if frame_dir is None:
        frame_dir = FRAME_DIR
    name = vid_pth.split('/')[-1].split('.')[0]

    frame_dir = os.path.join(frame_dir,name)
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    
    print('Start extraction from %s...'%vid_pth)
    cap = cv2.VideoCapture(vid_pth)
    cnt = 0
    while cap.isOpened():
        success,image = cap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(frame_dir,'%d.png'%cnt),image)
        
        cnt += 1
        if cnt%PRINT_STATUS_FREQUENCY == 0:
            print('Processed %d frames'%cnt)

        
    cap.release()

if __name__=="__main__":    
    extract_frames(args.video)