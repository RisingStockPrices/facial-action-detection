import numpy as np
from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms 

from emonet.models import EmoNet
from emonet.data import AffectNet
from emonet.data_augmentation import DataAugmentor
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from emonet.evaluation import evaluate, evaluate_flip

torch.backends.cudnn.benchmark =  True

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
args = parser.parse_args()

# Parameters of the experiments
n_expression = args.nclasses
batch_size = 32
n_workers = 16
device = 'cuda:0'
image_size = 256

# Loading the model 
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

# =======changin' ============
from skimage import io
import cv2
from torchvision import transforms
import PIL
import os

def tensor_to_image(tensor,out_dir='./test/results',prefix=None):
    if len(tensor.shape)==4:
        raise NotImplemented
    else:
        if prefix is not None:
            prefix = prefix+'_test'
        else:
            prefix = 'test'
        # choose filename
        idx = len([f for f in os.listdir(out_dir) if f.startswith(prefix)])
        fname = '%s_%d.png'%(prefix,idx)
        pth = os.path.join(out_dir,fname)

        if isinstance(tensor,np.ndarray):
            # import pdb;pdb.set_trace()
            cv2.imwrite(pth,tensor)
        else:
            image = torch.permute(tensor,(1,2,0))
            image = (255*image).cpu().numpy()
            
            cv2.imwrite(pth,image)

def get_aroval(pth_or_image,exp=True,val=True,aro=True):
    
    if isinstance(pth_or_image,str):
        image = cv2.imread(pth_or_image)


    image = pth_or_image
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
        ])

    try:
        image = transform(image)
    except:
        return None,None,None
    # tensor_to_image(image)
    image = torch.unsqueeze(image,dim=0)

    with torch.no_grad():
        # import pdb;pdb.set_trace()
        out = net(image)
        
        heatmap = out['heatmap']
        exp = out['expression']
        val = out['valence']
        aro = out['arousal']

    # import pdb;pdb.set_trace()

    return exp,val,aro

if __name__=="__main__":

    with torch.no_grad():
        image_file = '/home/spock-the-wizard/cmu/sg/visa_photo.jpg'

        get_aroval(image_file)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(256),
        # ])

        # image = cv2.imread(image_file)
        # image = transform(image)

        # tensor_to_image(image)

        # image = torch.unsqueeze(image,dim=0)
        
        # out = net(image)
    
        # print('Saved image')

        
        # heatmap = out['heatmap']
        # exp = out['expression']
        # val = out['valence']
        # aro = out['arousal']

        # print(out)


