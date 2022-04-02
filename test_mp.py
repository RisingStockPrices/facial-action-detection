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

def valaro_plot(val_lst,aro_lst,saveimg=False,num_records=5):
  
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
  # detection_results.append(bbox_points)

if __name__=="__main__":

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_face_detection = mp.solutions.face_detection #mesh = mp.solutions.face_mesh

  # For webcam input:
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

  vid_pth = '/home/spock-the-wizard/cmu/sg/emonet/test/data/wink_test.mp4'
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
  with mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
      cnt += 1
      if cnt > max_frames:
        break
      success, image = cap.read()
      if not success:
        # print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

      results = face_detection.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      image_org = image.copy()
      
      if results.detections:
        for detection in results.detections:
          mp_drawing.draw_detection(image,detection)
          # import pdb;pdb.set_trace()

          face = extract_face(image_org,detection.location_data.relative_bounding_box)

          # model.tensor_to_image(face,prefix='face')
          # import pdb;pdb.set_trace()

          exp,val,aro = model.get_aroval(face)

          if exp is None:
            val_lst.append(val_lst[-1])
            aro_lst.append(aro_lst[-1])
            face = np.zeros((400,400,3),dtype=np.uint8)
            # import pdb;pdb.set_trace()
          else:
            exp = torch.argmax(exp).item()
            val = val.item()
            aro = aro.item()

            val_lst.append(val)
            aro_lst.append(aro)

          plt_size = min(plot_size,frame_height)
          valaro = valaro_plot(val_lst,aro_lst)
          plots = cv2.vconcat([cv2.resize(face,(200,200)),cv2.resize(valaro,(200,200))])
          plots = pad_images_to_same_size(plots,(frame_height,plot_size))

          # model.tensor_to_image(plots)
          image = cv2.hconcat([image,plots])
          out.write(image)
          # model.tensor_to_image(image) #cv2.imwrite('./test/results/extended.png')
          # import pdb;pdb.set_trace()
          break    
          

  cap.release()
  out.release()
  # out_plot.release()