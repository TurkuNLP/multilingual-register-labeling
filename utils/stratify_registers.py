#!/usr/bin/python

import collections
import random
import sys
import csv


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--seed', default=None, type=int)
    ap.add_argument('--deviance', default=0.01, type=float)
    ap.add_argument('--dev_ratio', default=0.2, type=float)
    ap.add_argument('--test_ratio', default=0.2, type=float)
    ap.add_argument('data')
    return ap


def get_label_counts(dataset):
    label_counts = collections.Counter()
    for ex in dataset:
        for label in ex['labels'].split():
            label_counts[label] += 1
    return label_counts


data = []
csv.field_size_limit(10000000)

args = argparser().parse_args(sys.argv[1:])
if args.seed:
    random.seed(args.seed)

with open(args.data) as file:
    print("Reading file", args.data)
    for row in csv.reader(file, delimiter='\t'):
        id, url, labels, text = row
        data.append({'labels': labels, 'data': '\t'.join(row)})
    print("Number of lines:", len(data))


DEVIANCE_LIMIT = args.deviance

DEV_RATIO = args.dev_ratio
TEST_RATIO = args.test_ratio
#TRAIN_RATIO = 1.0-DEV_RATIO-TEST_RATIO


all_label_counts = get_label_counts(data)
print("Number of labels in corpus:", len(all_label_counts))
print()
assert min(all_label_counts.values()) >= 3

while True:
    print("Shuffling...")
    random.shuffle(data)
    ok = [False]*3
    max_deviance = 0
    for i, (name, beg, end) in enumerate([('dev', 0, int(len(data)*DEV_RATIO)),
                                          ('test', int(len(data)*DEV_RATIO), int(len(data)*(DEV_RATIO+TEST_RATIO))),
                                          ('train', int(len(data)*(DEV_RATIO+TEST_RATIO)), None)]):
        subset = data[beg:end]
        label_counts = get_label_counts(subset)
        print("  Subset:", name)
        print("    Number of labels:", len(label_counts))
        ok[i] = len(label_counts)==len(all_label_counts)
        rel_deviance = 0.
        for label, count in all_label_counts.most_common():
            rel_deviance += float(label_counts[label]/len(subset)-count/len(data))

        print("    Accumulated label frequency deviance: %.2f%%" % (rel_deviance*100))
        max_deviance = max(max_deviance, abs(rel_deviance))

    if all(ok):
        if max_deviance < DEVIANCE_LIMIT:
            print("Split is good! \\o/\n")
            break
    else:
        print("Split is no good :(\n")


print("--- Final result ---")
for i, (name, beg, end) in enumerate([('dev', 0, int(len(data)*DEV_RATIO)),
                                      ('test', int(len(data)*DEV_RATIO), int(len(data)*(DEV_RATIO+TEST_RATIO))),
                                      ('train', int(len(data)*(DEV_RATIO+TEST_RATIO)), None)]):
    subset = data[beg:end]
    label_counts = get_label_counts(subset)
    print("Subset:", name)
    print("Label distribution:")
    print('\t'.join(["%s (%.2f%%; %d)" % (label, label_counts[label]/len(subset)*100, label_counts[label]) for label, count in all_label_counts.most_common()]))
    out_name = 'stratified_%s_%s' % (name, args.data.split('/')[-1])
    print("Saving to %s" % out_name)
    with open(out_name, 'w') as f_out:
        for line in subset:
            print(line['data'], file=f_out)
    print()
