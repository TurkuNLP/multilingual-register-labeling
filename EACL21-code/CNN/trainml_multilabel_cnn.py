#!/usr/bin/env python3

# Train multilingual CNN text classifier.

import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import pickle

from logging import error

from common_multilabel import load_word_vectors_multi, load_fasttext_data_multi
from common_multilabel import texts_to_indices, label_to_index
from common_multilabel import transform2mlb
from common_multilabel import transform2mlb_val
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_curve,average_precision_score,auc
from joblib import dump, load

def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ag = ap.add_argument_group('required arguments')
    ag.add_argument('-i', '--input', metavar='LANG:PATH[,LANG:PATH...]',
                    required=True,
                    help='training data in language LANG (fastText format)')
    ag.add_argument('-o', '--output', metavar='MODEL', required=True,
                    help='model file path')
    ag.add_argument('-w', '--word-vectors', metavar='LANG:PATH[,LANG:PATH...]',
                    required=True,
                    help='word vectors for language LANG')
    ap.add_argument('-c', '--cased', default=False, action='store_true',
                    help='preserve input case (default: lowercase)')
    ap.add_argument('-d', '--dropout', metavar='FLOAT', default=None,
                    type=float, help='dropout rate')
    ap.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs to train for')
    ap.add_argument('-k', '--kernel-size', default=1, type=int,
                    help='convolution window length')
    ap.add_argument('-l', '--limit', default=None, type=int,
                    help='maximum number of word vectors to load per language')
    ap.add_argument('-L', '--max-length', default=5000, type=int,
                    help='maximum token sequence length')#1000
    ap.add_argument('-R', '--learning_rate', default=None, 
                    help='learning rate')
    ap.add_argument('-p', '--plot', default=False, action='store_true',
                    help='plot training history')
    ap.add_argument('-q', '--quiet', default=False, action='store_true',
                    help='suppress messages')
    ap.add_argument('-t', '--train-embedding', default=False,
                    action='store_true', help='trainable embeddings')
    ap.add_argument('-v', '--validation', metavar='LANG:PATH', default=None,
                    help='validation data')
    ap.add_argument('-r', '--predictions', default=None,
                    help="where predictions are stored")
    ap.add_argument('-T', '--threshold', default=0.5,
                    help='round to 1 or 0')
    return ap


def build_model(num_classes, embeddings, options): #def build_model(num_classes, embeddings, options):
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D #from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
    from keras.layers import Dropout

 #   print("building model", file=sys.stderr,flush=True)
    model = Sequential()
    vocab_size, embedding_dim = embeddings.shape[0], embeddings.shape[1]
    model.add(Embedding(
        vocab_size,
        embedding_dim,
        input_length=options.max_length,
        weights=[embeddings],
        trainable=options.train_embedding,
    ))
    model.add(Conv1D(128, options.kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    if options.dropout:
        model.add(Dropout(rate=options.dropout))
    print("num classes", num_classes, file=sys.stderr,flush=True)
    model.add(Dense(num_classes,activation='sigmoid'))
#    model.add(Dense(num_classes, activation='softmax')) #original
    return model


def parse_lang_path_argument(arg, sep=','):
    langs, paths = [], []
    for lang_path in arg.split(sep):
        try:
            lang, path = lang_path.split(':', 1)
        except:
            raise ValueError('missing ":" in LANG:PATH'.format(lang_path))
        if lang in langs:
            raise ValueError('repeated LANG value {}'.format(lang))
        langs.append(lang)
        paths.append(path)
    return langs, paths


def main(argv):
    args = argparser().parse_args(argv[1:])
    log = sys.stderr if not args.quiet else NullOutput()

    # Parse --word-vectors argument and load word vectors
    wv_langs, wv_paths = parse_lang_path_argument(args.word_vectors)
    embeddings, token_to_idx = load_word_vectors_multi( #embeddings, token_to_idx = load_word_vectors_multi(
        wv_langs, wv_paths, args.limit, log)
    print("word vecs loaded", file=sys.stderr, flush=True)

    # Parse --input argument and load texts and labels
    in_langs, in_paths = parse_lang_path_argument(args.input)
    if any(l for l in in_langs if l not in wv_langs):
        raise ValueError('missing --input LANG from --word-vectors')
    print("train from", in_langs, in_paths)
    texts, labels = load_fasttext_data_multi(
        in_langs, in_paths, not args.cased, log)
    if args.validation is not None:
        v_langs, v_paths = parse_lang_path_argument(args.validation)
        v_texts, v_labels = load_fasttext_data_multi(
            v_langs, v_paths, not args.cased, log)
    else:
        v_texts, v_labels = None, None

    # Map texts and labels to indices

    text_indices = texts_to_indices(texts, token_to_idx, False, 'train', log)
    print("text indices loaded", file=sys.stderr,flush=True)


    label_indices, mlb = transform2mlb(labels)
    dump(mlb,"mlb.joblib")
    
#    label_to_idx = {}
 #   label_indices = [label_to_index(l, label_to_idx) for l in labels]
  #  print("label_to_idx", label_to_idx)
   # print("labels", labels[:10])

    if args.validation is not None:
        print("indexing validation", file=sys.stderr, flush=True)
        v_text_indices = texts_to_indices(v_texts, token_to_idx, False,
                                          'validation', log)
        v_label_indices = transform2mlb_val(v_labels,mlb)
    num_classes=len(mlb.classes_)
    print('Loaded {} texts, {} labels'.format(
        len(text_indices), num_classes), file=log)
    lengths = [len(t) for t in text_indices]
    print('Text length min: {:.0f} max: {:.0f} avg: {:.0f}'.format(
        min(lengths), max(lengths), np.mean(lengths)), file=log)

    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from keras.optimizers import Adam
    
    train_X = pad_sequences(text_indices, maxlen=args.max_length,
                            padding='post', truncating='post')
    train_Y = label_indices#labels,num_classes #modified. this should do it bc we have labels binarized
    print('Train X shape: {}'.format(train_X.shape), file=log)
    print('Train Y shape: {}'.format(train_Y.shape), file=log)

    if args.validation is not None:
        valid_X = pad_sequences(v_text_indices, maxlen=args.max_length,
                                padding='post', truncating='post')
        valid_Y = v_label_indices
    model = build_model(num_classes, embeddings, args)
    lr = float(args.learning_rate)
    lr = float(lr)
    adam = Adam(lr=lr)
    model.compile(loss='binary_crossentropy',optimizer=adam)
    
    if args.validation is None:
        validation_data = None
    else:
        validation_data = (valid_X, valid_Y)
        print("validation x,y shape", valid_X.shape, valid_Y.shape, file=log,flush=True)
    history=model.fit(train_X,train_Y,batch_size=32,epochs=args.epochs,
                        validation_data=validation_data, shuffle=True)

    modelfn = '{}.h5'.format(args.output)
    metafn = '{}.json'.format(args.output)
    model.save(modelfn)
    print('Saved model in {}'.format(modelfn), file=log)
    with open(metafn, 'w', encoding='utf-8') as out:
        meta = {
            'cased': args.cased,
            'max_length': args.max_length,
#            'label_to_idx': label_indices,#label_to_idx,
            'token_to_idx': token_to_idx,
        }
        json.dump(meta, out, ensure_ascii=False, indent=4, sort_keys=True)
    print('Saved metadata in {}'.format(metafn), file=log)
    
    return 0


if __name__ == '__main__':
    try:
        retval = main(sys.argv)
    except ValueError as e:
        error(e)
        retval = -1
    sys.exit(retval)
