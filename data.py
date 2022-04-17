
"""
Class for loading data
"""
ACTION_LABELS = ['l-wink','lip-squeeze','eye-roll','chipmunk-face','nose-wrinkle','smirk','jawdrop','nostril-flex','r-wink','both-eyebrow-raise']

index2label = dict([(k,v) for k,v in enumerate(ACTION_LABELS)])
label2index  = dict([(v,k) for k,v in enumerate(ACTION_LABELS)])
face1_action_script = [(9,23),(41,58),(77,99),(144,168),(192,216),(237,258),(282,299),(362,383),(408,419),(457,473),(490,504),(524,541),(565,585),(602,616),(634,647),(673,681),(705,719),(746,758),(783,801),(817,831),(853,879)]
face2_action_script = [(19,30),(51,62),(73,86),(118,129),(156,166),(189,204),(231,243),(263,279),(301,311),(333,340),(365,386),(406,420),(427,452),(469,478),(504,511),(None,None),(637,651),(664,674),(731,744),(769,780),(809,833)]
action_script_labels = [0,1,2,3,4,9,5,6,7,8,1,3,9,5,0,7,4,8,1,9,3]

# # face1.mp4
# face1 = {
#     "l-wink":[(9,23)],
#     "lip-squeeze":[(41,58)],
#     "eye-roll":[(77,99),],
#     "chipmunk-face":[(41,58)],
#     'nose-wrinkle':[],'smirk','jawdrop','nostril-flex','r-wink','both-eyebrow-raise'
# }
def create_gt_action_dict(detection,labels):
    assert len(detection)==len(labels)

    action_script = dict()
    for range,label in zip(detection,labels):
        
        label = index2label[label]
        if label in action_script.keys():
            action_script[label].append(range)
        else:
            action_script[label] = [range]

    return action_script

ACTION_SCRIPT_FACE1 = create_gt_action_dict(face1_action_script,action_script_labels)
ACTION_SCRIPT_FACE2 = create_gt_action_dict(face2_action_script,action_script_labels)

if __name__=="__main__":
    print('='*50)
    print('face1.mp4 action script')
    
    print('='*50)
    print(ACTION_SCRIPT_FACE1)
    
    print('='*50)
    print('face2.mp4 action script')
    
    print('='*50)
    print(ACTION_SCRIPT_FACE2)

    # print('label mapping:',index2label)
    
