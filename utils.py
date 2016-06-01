import os as os
import numpy as np


def init_babi(path):
    print "==> Loading from %s" % path
    tasks = []
    task = None
    for fname in os.listdir(path):
        with open(os.path.join(path, fname)) as f:
            task = {"C": "", "Q": "", "A": ""}
            for i, line in enumerate(f):
                line = line.strip()
                if i == 2:
                    task["C"] = line + " "
                elif i == 4:
                    task["Q"] = line
                elif i == 6:
                    task["A"] = line
            tasks.append(task.copy())

    return tasks


def init_file(path):
    print "==> Loading from %s" % path
    tasks = []
    task = None
    with open(path) as f:
        task = {"C": "", "Q": "", "A": "", "E": {}}
        for i, line in enumerate(f):
            line = line.strip()
            if i == 2:
                task["C"] = line + " "
            elif i == 4:
                task["Q"] = line
            elif i == 6:
                task["A"] = line
            elif i > 7:
                task["E"][line.split(":")[0]] = line.split(":")[1]
        tasks.append(task.copy())

    return tasks

def get_babi_raw(dataset_name, index):
    if index != '':
        babi_train_raw = init_babi(os.path.join(
            os.path.dirname(os.path.realpath(
                __file__)), 'data/%s/questions/training_split/%s' % (dataset_name, index)))
    else:
        babi_train_raw = init_babi(os.path.join(
            os.path.dirname(os.path.realpath(
                __file__)), 'data/%s/questions/validation/' % dataset_name))
    babi_test_raw = init_babi(os.path.join(
        os.path.dirname(os.path.realpath(
            __file__)), 'data/%s/questions/test/' % dataset_name))
    return babi_train_raw, babi_test_raw


def load_glove(dim):
    word2vec = {}

    print "==> loading glove"
    with open(os.path.join(
            os.path.dirname(os.path.realpath(
                __file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])

    print "==> glove is loaded"

    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=False):
    # Create a vector corresponding to the @entity number
    try:
        entity_num = float(word[7:])
        vector = np.array(np.empty(word_vector_size))
        vector.fill(entity_num)
    # If word is not @entity create a random vector
    except ValueError:
        vector = np.array(np.empty(word_vector_size))
        vector.fill(0.5)
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector


def process_word(word,
                 word2vec,
                 vocab,
                 ivocab,
                 word_vector_size,
                 to_return="word2vec",
                 silent=False,
                 mode='train'):
    if not word in word2vec:
        if "@entity" in word:
            create_entities(word2vec, word_vector_size)
            create_vector(word, word2vec, word_vector_size, silent)
        else:
            # Use unknown token for all other words not in word2vec
            word = "@unknown"
            if not word in word2vec:
                create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab and "@entity" in word:
        vocab, ivocab = create_entities_vocab(vocab, ivocab)
        if mode == 'predict':
            vocab["."] = 1999
            ivocab[1999] = "."

    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def create_entities(word2vec, word_vector_size):
    for i in range(2000):
        vector = np.array(np.empty(word_vector_size))
        vector.fill(i)
        word2vec["@entity"+str(i)] = vector
    return vector


def create_entities_vocab(vocab, ivocab):
    for i in range(2000):
        vocab["@entity"+str(i)] = i
        ivocab[i] = "@entity"+str(i)
    return vocab, ivocab


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)
