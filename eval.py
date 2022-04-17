from data import *

from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_curve,roc_auc_score,RocCurveDisplay

# sklearn.metrics.roc_curve(y_true, y_score, 

def add_false_negatives(pred_list,gt_list):
    count = 0
    for gt in gt_list:
        # if no intersection with predictions
        # count as false negative
        if iou(gt,pred_list)==0:
            count += 1
    
    # gonna be used as input to roc_curve func
    y_true = [0,]*count
    y_pred = [0,]*count
    return y_true,y_pred

    
def iou(target,gt_list):
    start,end = target
    for from_,to_ in gt_list:
        # no overlap
        if end<=from_ or to_<=start:
            continue
        first = abs (to_-start)
        second = abs(from_-end) #union = to_-start
        if first<second:
            return first/second
        else:
            return second/first
    return 0

def compute_confidence_score(out,gt):
    res = []
    for o in out:
        score = iou(o,gt)
        assert score>=0 and score<=1
        res.append(score)
    return res

def preprocess_roc_input(out,gt):

    scores = compute_confidence_score(out,gt)
    y_pred = scores
    y_true = [0 if score==0 else 1 for score in scores]

    for gt_ in gt:
        # if no intersection with predictions
        # count as false negative
        if iou(gt_,out)==0:
            y_true.append(0)
            y_pred.append(0)
    
    return y_true,y_pred

def evaluate(out,gt,plot_fname=None,return_score=False):
    
    y_true,y_pred = preprocess_roc_input(out,gt)
    
    res = roc_curve(y_true,y_pred)
    score = roc_auc_score(y_true,y_pred)
    plot = RocCurveDisplay.from_predictions(y_true,y_pred)
    
    if plot_fname is not None:
        plot.figure_.savefig(plot_fname)
    if return_score:
        return res,score
    return res
    
if __name__=="__main__":
    out = [(9,12),(33,38),(50,59)]
    
    wink_gt = ACTION_SCRIPT_FACE1['l-wink']
    
    y_true,y_pred = preprocess_roc_input(out,wink_gt)
    res = roc_curve(y_true,y_pred)
    score = roc_auc_score(y_true,y_pred)

    plot = RocCurveDisplay.from_predictions(y_true,y_pred)
    
    import pdb;pdb.set_trace()
