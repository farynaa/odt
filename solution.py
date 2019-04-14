# ITS-partner Applied AI
# Farina Alexander

import f1scores


if __name__ == "__main__":
    # results will be stored as dictionary
    results = {}
    # this list is for raw f1-scores data to check it manually if needed
    raw_data = []
    # now we can chose number of iteration (precision) for calculating best threshold values
    n_iter = 50

    for i in range(n_iter):
        
        print('current threshold: ',i/n_iter)
        # getting f1 scores with current threshold
        f1_scores = f1scores.get_f1_scores(i/n_iter)
        # store raw data
        raw_data.append(list(f1_scores.values()))

        # let's check what we got for this iteration. If f1 scores for some classes are better, we'll update output dictionary
        for k,v in f1_scores.items():
            # initialize new classes
            if k not in results:
                results[k] = {'best_f1':0., 'conf_thr': 0.}
            # if f1 score for this iteration is better, adjust it
            if v > results[k]['best_f1']:
                results[k]['best_f1'] = v
                results[k]['conf_thr'] = i/n_iter

    print('OPTIMAL CONFIDENCE DETECTION THRESHOLDS:')
    for k,v in results.items():
        print(k,v['conf_thr'])

    with open('thresholds.txt','w') as fout:
        fout.write(str(results))


    #print([row[0] for row in raw_data])



