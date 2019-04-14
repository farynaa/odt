import iou

# function calculates f1 score based on given TP, FP and FN values
def f1(TP,FP,FN):
    
    # if NO true positives just return zero
    if TP == 0:
        return 0.

    # according to definitions:
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # that is how f1 score calculated:
    return 2 * precision * recall / (precision + recall)

# this function extract ground truth and detetected objects from a line in log-file
def parse_string(s):
    
    # results will be lists
    ground, detect = [], []

    try: # something can go wrong
        # splitting line as we know that ground truth and detected objects are separated by '--'
        spl = s.split('--')
        # let's take ground truth sub-string first:
        for i,el in enumerate(spl[0].split(';')):
            # split it to extract class and box 
            # example 'Blouse,212,40,32,100'           
            spl2 = el.split(',')
            class_name = spl2[0]
            # box will be an immutable tuple
            box = tuple(int(el) for el in spl2[1:5])
            # store each ground truth object record as dictionary
            ground.append({'class':class_name, 'box':box})
        # the same is for detected objects.
        if len(spl[1]) > 1: # sometimes no objects were detected
            for i,el in enumerate(spl[1].rstrip(';\n').split(';')): # each line ends with ';\n', we need to delete it
                spl2 = el.split(',')
                class_name = spl2[0]
                box = tuple(int(el) for el in spl2[1:5])
                # detected objects have confidence score, ground truth doesn't
                conf = float(spl2[5])
                detect.append({'class':class_name, 'box':box, 'conf':conf})
    except:
        return None, None
    
    return ground, detect

# simpe function that deletes detected objects if their confidence score lower than given
def filter_by_threshold(detect, thr):
    return [d for d in detect if d['conf'] > thr]

# optional function, that delete duplicate true postivies detections.
def remove_duplicates(ground, detect):

    for i,g in enumerate(ground):

        tmp = []

        for d in detect:

            overlap = iou.iou(g['box'],d['box'])

            if g['class'] == d['class'] and overlap > 0.5:
                tmp.append(d)

        if len(tmp) > 1:

            sorted_tmp = sorted(tmp, key=lambda x: x['conf'], reverse=True)

            for el in sorted_tmp[1:]:
                detect.remove(el)

    return ground, detect