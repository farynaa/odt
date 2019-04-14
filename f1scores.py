import utilities
import iou

# main function that parse log-file line by line and extracts TP, FP, FN for each class
def get_f1_scores(thr):

    # main parametres
    INPUT_FILE = 'detection_val_log.txt'
    IOU_THR = 0.5
    CONF_THR = thr

    # open log-file
    with open(INPUT_FILE) as fin:
        content = fin.readlines()

    # TP, FP and FN for each class will be stored in a dictionary
    # Example: {'Blouse':{'TP':20,'FP':2,'FN':1}}
    cm_data = {}

    for line in content:
        
        # get ground truth and detected objects as lists
        ground, detect = utilities.parse_string(line)

        # filtering detected objects in respect of given confidence threshold
        detect = utilities.filter_by_threshold(detect, CONF_THR)

        # skip the line if it has wrong format
        if ground == None:
            continue

        # let check what classes are in this line and initialize output record with zeroes if the class is not previously seen
        for i in [el['class'] for el in ground + detect]:
            if i not in cm_data:
                cm_data[i] = {'TP':0, 'FP':0, 'FN':0}

        # initialize the main list of object comparison with zeroes
        matches = [[0 for i in range(len(ground))] for j in range(len(detect))]
        
        # fill in the list with 1 or 0 for each ground truh - detection pair
        for i,g in enumerate(ground):
            for j,d in enumerate(detect):
                # true positive means that classes are equal and boxes overlap is enough                
                if g['class'] == d['class'] and iou.iou(d['box'],g['box']) > IOU_THR:
                    matches[j][i] = 1
                else:
                    matches[j][i] = 0
        # checking our match-table for TP and FP
        for i,row in enumerate(matches):
            class_name = detect[i]['class']
            if sum(row) != 0:
                cm_data[class_name]['TP'] += 1
            else:
                cm_data[class_name]['FP'] += 1
        # checking it column-wise for FN
        for i in range(len(ground)):
            class_name = ground[i]['class']
            column = [row[i] for row in matches]
            if sum(column) == 0:
                cm_data[class_name]['FN'] += 1
    # storage for f1-scores
    f1_scores = {}
    
    # cm_data contains TP, FP, FN values for each class, but we need to calculate f1 scores.
    for k,v in cm_data.items():
        f1_scores[k] = utilities.f1(v['TP'],v['FP'],v['FN'])

    return f1_scores


if __name__ == "__main__":
    print(get_f1_scores(0.5))