import numpy as np

from collections import OrderedDict


def load_word_vectors(path, limit=None, prefix='', base_index=0):
    # Modified from
    # https://github.com/facebookresearch/MUSE/blob/amaster/demo.ipynb
    vectors = []
    token_to_idx = {}
    with open(path, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            word, vector = line.rstrip().split(' ', 1)
            word = prefix + word
            vector = np.fromstring(vector, sep=' ')
            if word in token_to_idx:
                raise ValueError('duplicate word {} in {}'.format(word, path))
            vectors.append(vector)
            token_to_idx[word] = base_index + len(token_to_idx)
            if limit is not None and len(token_to_idx) >= limit:
                break
    embeddings = np.vstack(vectors)
    return embeddings, token_to_idx


def load_word_vectors_multi(langs, paths, limit, log):
    # Load word vectors, add language prefixes to tokens, and return combined
    wordvec_data, base_index = OrderedDict(), 0
    if len(langs) != len(paths):
        raise ValueError('number of languages {} != number of paths {}'.format(
            len(langs), len(paths)))
    for lang, path in zip(langs, paths):
        if lang in wordvec_data:
            raise ValueError('repeated language {}'.format(lang))
        prefix = lang + ':'
        embeddings, token_to_idx = load_word_vectors(
            path, limit, prefix, base_index)
        print('loaded {} vectors from {}'.format(len(token_to_idx), path),
              file=log)
        wordvec_data[lang] = (embeddings, token_to_idx)
        base_index += len(token_to_idx)
    embeddings = np.concatenate([d[0] for d in wordvec_data.values()])
    token_to_idx = merge_dicts([d[1] for d in wordvec_data.values()])
    return embeddings, token_to_idx


def load_fasttext_data(fn, prefix='', lowercase=True, label_string='__label__'):
    texts, labels = [], []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split()
            label, tokens = fields[0], fields[1:]
            if not label.startswith(label_string):
                raise ValueError('no label on line {} in {}'.format(ln, fn))
            label = label[len(label_string):]
            if any(t for t in tokens if t.startswith(label_string)):
                raise ValueError('extra label on line {} in {}'.format(ln, fn))
            if lowercase:
                tokens = [t.lower() for t in tokens]
            tokens = [prefix+t for t in tokens]
            texts.append(' '.join(tokens))
            labels.append(label)
    return texts, labels


def load_fasttext_data_multi(langs, paths, lowercase, log):
    if len(langs) != len(paths):
        raise ValueError('number of languages {} != number of paths {}'.format(
            len(langs), len(paths)))
    texts_and_labels = OrderedDict()
    for lang, path in zip(langs, paths):
        if lang in texts_and_labels:
            raise ValueError('repeated language {}'.format(lang))
        prefix = lang + ':'
        texts, labels = load_fasttext_data(path, prefix, lowercase)
        print('loaded {} texts from {}'.format(len(texts), path), file=log)
        texts_and_labels[lang] = (texts, labels)
    texts = [t for d in texts_and_labels.values() for t in d[0]]
    labels = [l for d in texts_and_labels.values() for l in d[1]]
    return texts, labels


def text_to_indices(text, token_to_idx, extend_mapping=True):
    sequence = []
    for token in text.split(' '):
        if token in token_to_idx:
            sequence.append(token_to_idx[token])
        elif extend_mapping:
            token_to_idx[token] = len(token_to_idx)
            sequence.append(token_to_idx[token])
        else:
            text_to_indices.oov_count += 1
        text_to_indices.total_count += 1
    return sequence
text_to_indices.total_count = 0
text_to_indices.oov_count = 0


def texts_to_indices(texts, token_to_idx, extend_mapping, label, log):
    text_to_indices.total_count = 0
    text_to_indices.oov_count = 0
    indices = []
    for t in texts:
        indices.append(text_to_indices(t, token_to_idx, False))
    total, oov = text_to_indices.total_count, text_to_indices.oov_count
    print('Mapped {} data, OOV {}/{} {:.1%}'.format(
        label, oov, total, oov/total), file=log)
    return indices
                    

def label_to_index(label, label_to_idx):
    if label not in label_to_idx:
        label_to_idx[label] = len(label_to_idx)
    return label_to_idx[label]


def merge_dicts(dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


class NullOutput(object):
    def write(self, *args):
        pass
