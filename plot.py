import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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


def draw_1d_plot(title,array,fname=None,outdir='./test/results/plots',save=False):

    if fname==None:
        idx = len([f for f in os.listdir(outdir) if f.startswith('test')])
        fname = 'test_%d.png' % idx
    pth = os.path.join(outdir,fname)

    fig,ax = plt.subplots(figsize=(4,4))
    ax.plot(array)
    plt.title(title)
    if save:
        plt.savefig(pth)
        
    
    return fig,ax