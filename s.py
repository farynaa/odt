import iou

INPUT_FILE = 'detection_val_log.txt'

with open(INPUT_FILE) as fin:
    content = fin.readlines()

TP, FP, FN = 0, 0, 0

total_classes = {}

for l in content[:]:
    #print(l)
    real = {}
    detected = {}
    det_thr = {}
    spl = l.rstrip('\n').split('--')
    for el in spl[0].split(';'):
        s = el.split(',')
        label = s[0]
        coords = tuple([int(i) for i in s[1:]])
        real[label] = coords

    for el in spl[1].split(';')[:-1]:
        s = el.split(',')
        label = s[0]
        coords = tuple([int(i) for i in s[1:5]])
        if float(s[5]) > 0.3:
            detected[label] = coords
            det_thr[label] = s[5]

    for k in set(list(real.keys()) + list(detected.keys())):
        if k not in total_classes:
            total_classes[k] = 0.
    #print(real,detected)
    #print('======')
    #print(real, detected)

    for k in detected:
        if k not in real:
            FP += 1
        else:
            #print(k,(detected[k],real[k]), iou.iou(detected[k],real[k]))
            if iou.iou(detected[k],real[k]) > 0.5:
                TP += 1
            else:
                FP += 1
    for k in real:
        tmp = [el for el in detected if el in k and iou.iou(real[k],detected[el])]
        #print(len(tmp))
        if len(tmp) == 0:
            FN += 1

    #tmp = [(el,real[el]) for el in detected if el in real and iou.iou(detected[el],real[el]) > 0.5] # and iou.iou(real[k],detected[k])>0.5]
    #tmp2 = [iou.iou(detected[el],real[el]) for el in detected if el in real]
    #tmp2 = [(el, real[el]) for el in detected if el in real]

    #print(tmp2)
    #print(tmp)

        #if len(tmp) == 0:
        #    print(tmp)
precision = TP / (TP + FP)
recall = TP / (TP + FN)

f1 = 2 * precision * recall / (precision + recall)
print(total_classes)
print(len(total_classes))
print(precision,recall)
print('f1: ', f1)
print(FP,TP,FN)

