import os
import cv2

FRAME_DIR = './data/frames/'
def extract_frames(vid_pth,frame_dir=None):
    if frame_dir is None:
        frame_dir = FRAME_DIR
    name = vid_pth.split('/')[-1].split('.')[0]

    frame_dir = os.path.join(frame_dir,name)
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    cap = cv2.VideoCapture(vid_pth)
    cnt = 0
    while cap.isOpened():
        success,image = cap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(frame_dir,'%d.png'%cnt),image)
        
        # import pdb;pdb.set_trace()
        cnt += 1
        
    cap.release()

if __name__=="__main__":
    # import pdb;pdb.set_trace()
    pth = './data/faces/test2.mp4'
    extract_frames(pth)