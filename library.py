import pandas as pd
import numpy as np

import os.path as osp

from aus import *

DEFAULT_NAME = 'unnamed'
LIB_ROOT = './data/action_lib'

class FacialActionLibrary:
    def __init__(self,root=LIB_ROOT):
        self.lib_root = root
        self.aus_root = osp.join(root,'aus')

        self.lib_csv = osp.join(root,'lib.csv')

        if osp.exists(self.lib_csv):
            self.library = pd.read_csv(self.lib_csv)
        else:
            self.library = pd.DataFrame(columns=['aus_file_path','source_vid','start_idx','end_idx','nickname'])
            self.save_library()

    def save_library(self):
        self.library.to_csv(self.lib_csv,index=False)

    def register(self,aus,src,range,nickname=None):
        aus = aus.to_numpy()
        aus = aus[range[0]:range[1]]

        idx = len(self.library)
        # basic schema violation check
        if not isinstance(nickname,str):
            nickname = DEFAULT_NAME
        else:
            if len(self.query_by_nickname(nickname))!=0:
                print('Nickname already exists in library')
                raise KeyError
        if len(range)!=2:
            print('range should be a tuple')
            raise TypeError
        if not src.endswith('.avi') and not src.endswith('.mp4'):
            print('source should be a video file path')
            raise TypeError
            
        import pdb;pdb.set_trace()

        # save numpy file
        aus_file = osp.join(self.aus_root,'%d.npy'%(idx))
        np.save(aus_file,aus)

        # add to dataset
        self.library.loc[idx] = [aus_file,src,range[0],range[1],nickname]
        self.save_library() 
        
        print('Successfully registered %s in library' % aus_file)

    def query_by_nickname(self,nickname):
        df = self.library.loc[self.library['nickname']==nickname]
        return df
       

    def retrieve_aus(self,nickname):

        if nickname==DEFAULT_NAME or not isinstance(nickname,str):
            raise TypeError
        df = self.query_by_nickname(nickname)
        if len(df)!=1:
            print('Not a unique nickname')
            raise KeyError
        # import pdb;pdb.set_trace()
        aus = np.load(df.iloc[0]['aus_file_path'])
        return aus
    
    def delete_entry(self,nickname):
        if nickname is DEFAULT_NAME:
            print('Cant search default name %s'%DEFAULT_NAME)
            return None
        df = self.query_by_nickname(nickname)
        if len(df)==0:
            print('No entry found with name %s'%nickname)
        elif len(df)>1:
            print('found multiple items with nickname... raising error')
            raise KeyError
        else:
            # unique entry
            resp = input('Sure you wanna delete %s?[Y/n]'%nickname)
            if resp=='Y' or resp=='y':
                self.library = self.library.drop(df.index)
                self.save_library() 
                print('Successfully deleted %s'%nickname)
            else:
                return
            
        

FAL = FacialActionLibrary()

if __name__=="__main__":
    video_name='face1.mp4'
    src_vid = osp.join(SRC_DIR,video_name)
    aus = extract_aus_from_video(src_vid)[:-1]
    action_range = (9,23)
    FAL.register(aus,video_name,action_range,'l_wink')
