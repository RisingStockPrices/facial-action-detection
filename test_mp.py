import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import model
import os

# constants
expression_labels = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}
pannel_size = 100

# hyperparams
n_expression = 8 #args.nclasses
image_size = 256
device = 'cuda:0'

def pad_images_to_same_size(img,size):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """

    hh,ww = size
    mins = min(hh,ww)
    img = cv2.resize(img,(mins,mins))
    # min(hh)
    # import pdb;pdb.set_trace()
    h, w = img.shape[:2]
    diff_vert = hh - h
    pad_top = diff_vert//2
    pad_bottom = diff_vert - pad_top
    diff_hori = ww - w
    pad_left = diff_hori//2
    pad_right = diff_hori - pad_left
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    assert img_padded.shape[:2] == (hh,ww)
    # images_padded.append(img_padded)

    return img_padded

def valaro_plot(val_lst,aro_lst,saveimg=True,num_records=5):
  
    # leave only last k points
    
    val_lst = val_lst[-num_records:]
    aro_lst = aro_lst[-num_records:]

    fig,ax = plt.subplots(figsize=(4,4))
    # fig.add_axes([0.1,0.1,0.8,0.8],polar=True)




    
    ax.plot(val_lst,aro_lst,marker='o',
     markerfacecolor='blue', markersize=2)
    # plt.scatter(val_lst,aro_lst,color='green',s=2)
    plt.title('Valence-Arousal Plot')
    plt.xlabel('valence')
    plt.ylabel('arousal')
    plt.axhline(0, lw=0.5, color='black')
    plt.axvline(0, lw=0.5, color='black')
    # plt.legend()

    plt.xlim([-1,1])
    plt.ylim([-1,1])
    
    # ax.set_prop_cycle(color=['red','orange','yellow','green','blue'])
    
    # import pdb;pdb.set_trace()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # import pdb;pdb.set_trace()
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    if saveimg:
      # optionally save name with prefix (not implemented yet)
      prefix = 'valaro'
      plt.savefig('tmp.png',transparent=True)

    plt.close(fig)
    # import pdb;pdb.set_trace()
    return img

# # Loading the model 
# state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

# print(f'Loading the model from {state_dict_path}.')
# state_dict = torch.load(str(state_dict_path), map_location='cpu')
# state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
# net = EmoNet(n_expression=n_expression).to(device)
# net.load_state_dict(state_dict, strict=False)
# net.eval()



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

vid_pth = '/home/spock-the-wizard/cmu/sg/emonet/test/data/100teens.mp4'
cap = cv2.VideoCapture(vid_pth)
plot_size = 400
frame_width = int(cap.get(3)+plot_size)
frame_height = int(cap.get(4))

vidname = vid_pth.split('/')[-1].split('.')[0]
outdir = '/home/spock-the-wizard/cmu/sg/emonet/test/results'
outpth = os.path.join(outdir,'out_%s.avi'%vidname)
out = cv2.VideoWriter(outpth,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# outpth_plot = outpth.replace('out','out_plot')
# out_plot = cv2.VideoWriter(outpth_plot,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (400,400))

# import pdb;pdb.set_trace()
cnt = 0
max_frames = 1000
val_lst = []
aro_lst = []
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    cnt += 1
    if cnt > max_frames:
      break
    success, image = cap.read()
    if not success:
      # print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        import pdb;pdb.set_trace()

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

        # get bounding box (only face part)
        # 
        # put that in the network
        # 
        # create plots
        # 
              
        # inference here
        exp,val,aro = model.get_aroval(image)

        exp = torch.argmax(exp).item()
        val = val.item()
        aro = aro.item()

        val_lst.append(val)
        aro_lst.append(aro)


        plt_size = min(plot_size,frame_height)
        black = np.zeros((frame_height,plt_size,3))
        img_plot = valaro_plot(val_lst,aro_lst)
        img_plot = pad_images_to_same_size(img_plot,(frame_height,plt_size))

        cv2.putText(image,'exp:%s'%(expression_labels[exp]),(10,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,255,255))
        cv2.putText(image,'val:%4f aro:%4f'%(val,aro),(10,100),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(255,255,255))




        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        image = cv2.hconcat([image,img_plot])
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # model.tensor_to_image(image) #cv2.imwrite('./test/results/extended.png')
        # import pdb;pdb.set_trace()
        # get top single face
        break


        out.write(image)
        

cap.release()
out.release()
# out_plot.release()