import pandas as pd
import os.path as osp
import numpy as np

import ruptures as rpt
import argparse

from library import *

PLOT_DIR = './data/cpd'

# parser = argparse.ArgumentParser()
# parser.add_argument('--plot_name')
# args = parser.parse_args()


def convert_to_flow_frames(aus,save_pth=None):
    """
    aus [pd.DataFrame]
    """

    # import pdb;pdb.set_trace()
    if pd.isna(aus.iloc[-1][0]):
        aus = aus[:-1]

    PENALTY = 0.3
    KERNEL = "rbf"
    result = rpt.Pelt(model=KERNEL,min_size=1,jump=1).fit_predict(aus,pen=PENALTY)
    segments = np.digitize(range(len(aus)),result)
    
    aus['segment'] = segments.copy()
    group_aus = aus.groupby("segment").mean()
    grouped = group_aus
    
    result.insert(0,0)
    mapping = [(a,b) for a,b in zip(result[:-1],result[1:])]
    # aus = aus.groupby('segment')
    # print(res)
    if save_pth:
        group_aus['start_idx'] = result[:-1]
        group_aus['end_idx'] = result[1:]
        group_aus.to_csv(save_pth)
    
    # import pdb;pdb.set_trace()
    return grouped,mapping



if __name__=="__main__":
    aus_file = './data/aus/face1.csv'
    import pdb;pdb.set_trace()
    signal = pd.read_csv(aus_file)
    aus_action = FAL.retrieve_aus('l_wink',return_df=True)
    
    aus, mapping = convert_to_flow_frames(signal)
    aus_action, mapping_action = convert_to_flow_frames(aus_action)

    import pdb;pdb.set_trace()

    # result = rpt.Pelt(model="rbf").fit_predict(signal,pen=0.2)
    # result = algo.predict(pen=10)
    
    # plot = rpt.display(signal,[], result)
    # pth = osp.join(PLOT_DIR,args.plot_name)
    # plot[0].savefig(pth)
    # import pdb;pdb.set_trace()