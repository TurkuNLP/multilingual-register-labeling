import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from logging import error
import sys, os, glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def map_to_label(fn, labels):
    multilabels = []
    for row in fn:
        label_names = []
        for index, label in enumerate(row):
            if label == 1:
                label_names.append(labels[index])
        multilabels.append(''.join(label_names) if len(label_names) <= 1 else 'HYB')
    return multilabels

def trim_classes(matrix, in_labels, preserved):
    m = matrix
    labels = in_labels
    hybrids = []
    
    for i, l in enumerate(labels):
        if "_" in l:
            hybrids.append(i)

    if len(m.shape) != 2 or m.shape[0] != m.shape[1]:
        error('expected a square 2d matrix, got {}'.format(m.shape))
        return 1

    dim = m.shape[0]
    rows = [0] * dim

    if type(labels) != list or len(labels) != dim:
        error('expected list of {} labels, got {}'.format(dim, labels))
        return 1

    colsum = m.sum(axis=0)
    rowsum = m.sum(axis=1)
    #print(f"colsum: {colsum}")

    delete_indices = []
    for i in range(dim):
        if i not in preserved and i not in hybrids:
            delete_indices.append(i)

    if len(delete_indices) == 0:
        error('no rows/cols deleted (try increasing threshold)')
        return 1
    if len(delete_indices) == dim:
        error('all rows/cols deleted (try decreasing threshold)')
        return 1
    print('deleting {}/{} rows and columns'.format(len(delete_indices), dim),
          file=sys.stderr)

    # reversed so indexing doesn't change on delete
    for i in reversed(delete_indices):
        m = np.delete(m, i, axis=0)
        m = np.delete(m, i, axis=1)
        del labels[i]

    #print(m)
    print('\t'.join([''] + labels))
    for i, row in enumerate(m):
        print('\t'.join([labels[i]] + [str(j) for j in row]))
    nM = []
    #return m, labels
    for i, row in enumerate(m):
        rsum = sum(row)
        print(f"Row is: {row} summing to {rsum}")
        newRow = []
        for j, item in enumerate(row):
            ratio = m[i][j] / rsum
            print(f"Calculating ratio: {m[i][j]}/ {rsum}")

            if np.isnan(ratio):
                print(m[i][j])
                newRow.append(rsum)
            else:
                newRow.append(ratio)
        nM.append(newRow)

    #print(nM)
    return nM, labels

for t, lang in enumerate(['en-fi', 'en-fr', 'en-sv', 'fi-fi', 'fr-fr', 'sv-sv']):
    language_pairs = ['English-Finnish', 'English-French', 'English-Swedish', 'Finnish-Finnish', 'French-French', 'Swedish-Swedish']
    gold_l = []
    pred_l = []
    for k in [1,2,3]:
        gold = np.load(f"numbers/{lang}-{k}.gold.npy", allow_pickle=True)
        pred = np.load(f"numbers/{lang}-{k}.preds.npy", allow_pickle=True)
        labels = np.load(f"numbers/{lang}.labels.npy", allow_pickle=True)

        gold_labels = map_to_label(gold, labels)
        pred_labels = map_to_label(pred, labels)

        gold_l.extend(gold_labels)
        pred_l.extend(pred_labels)
  
    multilabels = sorted(list(set(gold_l + pred_l)))
    multilabels.append(multilabels.pop(multilabels.index("HYB")))
 
    cf = confusion_matrix(y_true=gold_l, y_pred=pred_l, labels=multilabels)

    labels_to_preserve = ["HI", "ID", "IN", "IP", "NA", "OP", "HYB"]
    preserve_indices = []

    for label in labels_to_preserve:
        preserve_indices.append(multilabels.index(label))
    
    pres, prlabels = trim_classes(cf, multilabels, preserve_indices)

    maxim = (np.amax(pres))
    plt.figure(figsize=(12,12))
    heatmap = sns.heatmap(pres, annot=True, annot_kws={"size": 35}, cmap='Purples', xticklabels=prlabels, yticklabels=prlabels, fmt='.2f', robust=True, 
                    linewidths=0.5, vmax=maxim, square=True, cbar=False)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=35)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=35)

    heatmap.xaxis.tick_top() # x axis on top
    heatmap.xaxis.set_label_position('top')
    heatmap.tick_params(length=0)
    #plt.title(label=lang, fontweight=400, y=-0.01)
    plt.text(3.0, 7.25, language_pairs[t], fontsize=35, horizontalalignment='center', verticalalignment='center', weight='bold')
    #plt.title(language_pairs[t], fontsize=35)
    plt.tight_layout()
    plt.show()
  
