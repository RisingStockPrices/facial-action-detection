import matplotlib.pyplot as plt
from feat.plotting import plot_face
import os.path as osp
import os

def plot_aus(aus,pth='aus_tmp.png'):
    res = aus.plot()
    res.figure.savefig(pth)

def color_matches(aus,res,duration=1,pth='color_test.png',title='Detection Result'):
    # fig = plt.figure()
    plot = aus.plot()
    # n_res = len(res)
    for r in res:
        from_ = r[0]
        if len(r) == 2:
            to = from_+duration
        else:
            to = r[1]
        # attenuate colors
        alpha = 0.5-r[-1]
        
        # import pdb;pdb.set_trace()
        plot.axvspan(from_,to, color='red', alpha=alpha)
    
    plot.set_title(title)
    plot.figure.savefig(pth)

def test_face_plotter(aus):
    import pdb;pdb.set_trace()
    face1 = plot_face(model=None,au=3*aus[336],color='k', linewidth=1, linestyle='-')
    
    face2 = plot_face(model=None,au=3*aus[364],color='k', linewidth=1, linestyle='-')
    face1.figure.savefig('face1.png')
    face2.figure.savefig('face2.png')

def save_detection_results(video_aus,res,video,target_action,threshold,plot_dir='./data/aus_plot',):
    
    video = video.split('.')[0] # extract only name
    src_dir = osp.join(plot_dir,video)
    if not osp.exists(src_dir):
        os.mkdir(src_dir)
    
    fpath = osp.join(src_dir,'action:%s_threshold:%f.png'%(target_action,threshold))
    color_matches(video_aus,res,pth=fpath,title='action: %s, threshold: %f'%(target_action,threshold))
    