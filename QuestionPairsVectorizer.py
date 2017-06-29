# -*- coding: utf-8 -*-
'''
Определяем вспомогательные классы для векторизации списка пар вопросов.
Используется скриптами сеточных моделей.
'''

from __future__ import print_function

import os
import codecs
import collections
import itertools
import cPickle
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


class _BaseVectorizer:
    _data_folder = './data'


class CharVectorizer(_BaseVectorizer):
    """
    Векторизация пар вопросов для char-level моделей.
    """

    def _limit_chars(self, s):
        return s[:self._max_chars_per_line]

    def _load_qlist(self, filename):
        with open(os.path.join(self._data_folder, filename), "rb") as rdr:
            qlist = cPickle.load(rdr)
            return [self._limit_chars(s) for s in qlist]

    def __init__(self, max_chars_per_line=200):
        self._max_chars_per_line = max_chars_per_line
        self._char2freq = collections.Counter()

    def _load_data(self, csv_filepath):
        data = pd.read_csv(csv_filepath, delimiter=',', quoting=True, encoding='utf-8')
        data.fillna(value=u'', inplace=True)
        questions1 = data.question1.astype(unicode).values
        questions2 = data.question2.astype(unicode).values
        return (questions1,questions2)

    def get_train_char_vectorization(self, swap_questions, re_weight ):
        """
        Создание тензоров для обучения модели.
        Все необходимые исходные датасеты подгружаются из файлов именно в этом методе, поэтому
        ожидается большая загрузка кучи.
        :param swap_questions: удваивать датасет с помощью перестановок вопросов в парах
        :param re_weight:  учитывать разную долю дубликатов в тренировочном наборе и в данных для сабмита
        :return: структуру с подготовленными матрицами TODO describe more...
        """
        # <editor-fold desc="Загрузка датасетов">
        # questions1 = load_wordlist('train_tokens1.pkl')
        # questions2 = load_wordlist('train_tokens2.pkl')
        #questions1 = self._load_qlist('train_questions1.pkl')
        #questions2 = self._load_qlist('train_questions2.pkl')

        (questions1,questions2) = self._load_data( './data/train.csv' )
        (self.queries1,self.queries2) = self._load_data('./data/test.csv')

        ntrain = len(questions1)

        #self.queries1 = self._load_qlist('submit_queries1.pkl')
        #self.queries2 = self._load_qlist('submit_queries2.pkl')
        # queries1 = load_wordlist('submit_tokens1.pkl')
        # queries2 = load_wordlist('submit_tokens2.pkl')

        #is_dup = np.load(os.path.join(self._data_folder, 'y_train.npy'))
        data = pd.read_csv('./data/train.csv', delimiter=',', quoting=True, encoding='utf-8')
        is_dup = data.is_duplicate.values.astype('bool')
        # </editor-fold>

        # <editor-fold desc="Алфавит">
        self._all_chars = set()
        self._max_char_len = 0

        # построим список всех используемых символов
        self._all_chars = collections.Counter()
        self._max_char_len = 100
        for q in itertools.chain(questions1, questions2, self.queries1, self.queries2):
            qq = q[:self._max_char_len]
            self._all_chars.update(qq)
            self._char2freq.update(qq)

        self._char2index = {u'\0': 0}
        for (i, c) in enumerate(self._all_chars):
            self._char2index[c] = i + 1

        self._nb_chars = len(self._char2index)
        # </editor-fold>

        # <editor-fold desc="Векторизация обучающей матрицы">
        nb_patterns = ntrain
        X1_data = np.zeros((nb_patterns, self._max_char_len), dtype=np.int16) # предполагаем, что разных символов < 65k
        X2_data = np.zeros((nb_patterns, self._max_char_len), dtype=np.int16)
        y_data = np.zeros(nb_patterns, dtype=np.bool)

        idata=0
        for i,(q1,q2,y) in enumerate(zip(questions1, questions2, is_dup)[:nb_patterns]):
            y_data[idata] = bool(y)
            qq1 = q1[:self._max_char_len]
            qq2 = q2[:self._max_char_len]
            for j,c in enumerate(qq1):
                X1_data[idata,j] = self._char2index[c]

            for j, c in enumerate(qq2):
                X2_data[idata, j] = self._char2index[c]

            idata += 1

        VALIDATION_SPLIT = 0.1
        perm = np.random.permutation(nb_patterns)
        idx_train = perm[:int(nb_patterns * (1 - VALIDATION_SPLIT))]
        idx_val = perm[int(nb_patterns * (1 - VALIDATION_SPLIT)):]

        if swap_questions:
            # удваиваем датасет, переставляя местами вопросы в парах
            X1_train = np.vstack((X1_data[idx_train], X2_data[idx_train]))
            X2_train = np.vstack((X2_data[idx_train], X1_data[idx_train]))
            y_train = np.concatenate((y_data[idx_train], y_data[idx_train]))

            X1_val = np.vstack((X1_data[idx_val], X2_data[idx_val]))
            X2_val = np.vstack((X2_data[idx_val], X1_data[idx_val]))
            y_val = np.concatenate((y_data[idx_val], y_data[idx_val]))
        else:
            X1_train =X1_data[idx_train]
            X2_train = X2_data[idx_train]
            y_train = y_data[idx_train]

            X1_val = X1_data[idx_val]
            X2_val = X2_data[idx_val]
            y_val = y_data[idx_val]

        weight_val = np.ones(len(y_val))
        if re_weight:
            weight_val *= 0.472001959
            weight_val[y_val == 0] = 1.309028344

        if re_weight:
            class_weight = {0: 1.309028344, 1: 0.472001959}
        else:
            class_weight = None
        # </editor-fold>

        CharTrainingDataset = collections.namedtuple('CharTrainingDataset',
                                                     ['X1_train', 'X2_train', 'y_train',
                                                      'X1_val', 'X2_val', 'y_val',
                                                      'weight_val', 'class_weight',
                                                      'nb_chars', 'max_len'])

        return CharTrainingDataset(X1_train, X2_train, y_train,\
                                   X1_val, X2_val, y_val,\
                                   weight_val, class_weight,
                                   self._nb_chars, self._max_char_len)

    def get_submit_char_vectorization(self):
        """
        Подготовка матриц, по которым ранее обученная модель будет генерировать вероятности для сабмита.
        :return: (X1_submit,X2_submit)
        """
        ntest = len(self.queries1)

        X1_submit = np.zeros((ntest, self._max_char_len), dtype=np.int16)
        X2_submit = np.zeros((ntest, self._max_char_len), dtype=np.int16)

        for i in range(ntest):
            q1 = self.queries1[i][:self._max_char_len]
            for j, c in enumerate(q1):
                if c in self._char2index:
                    X1_submit[i, j] = self._char2index[c]

            q2 = self.queries2[i][:self._max_char_len]
            for j, c in enumerate(q2):
                if c in self._char2index:
                    X2_submit[i, j] = self._char2index[c]

        del self.queries1
        del self.queries2
        return (X1_submit, X2_submit)

    def store_stats(self):
        if len(self._char2freq) > 0:
            with codecs.open(os.path.join(self._data_folder, 'char2freq.dat'), 'w', 'utf-8') as wrt:
                for c, cnt in sorted(self._char2freq.iteritems(), key=lambda z: -z[1]):
                    wrt.write(u'{}\t{}\n'.format(c, cnt))


class WordVectorizer(_BaseVectorizer):
    """
    Векторизация пар вопросов для word-level моделей
    """
    
    #_w2v_path = 'w2v.CBOW=1_WIN=5_DIM=32.model'
    #binary_w2v = True

    _w2v_path = os.path.join( _BaseVectorizer._data_folder, 'glove.840B.300d.txt' )
    binary_w2v = False

    _ONLY_W2V_WORDS = False

    def _limit_tokens(self,tokens):
        return tokens[:self._max_words_per_line]

    def _load_words(self, filename):
        with open(os.path.join(self._data_folder, filename), 'rb') as rdr:
            tlist = cPickle.load(rdr)
            return [self._limit_tokens(tx) for tx in tlist]

    def __init__(self, max_words_per_line=40):
        self._max_words_per_line = max_words_per_line
        self._token2freq = collections.Counter()


    def _load_embeddings(self):
        if self.binary_w2v:
            self._w2v = gensim.models.KeyedVectors.load_word2vec_format(self._w2v_path, binary=True)
            self._WORD_DIMS = len(self._w2v.syn0[0])
        else:
            self._w2v = dict()
            self._WORD_DIMS = -1
            with open(self._w2v_path) as f:
                for line in f:
                    values = line.split()
                    word = values[0]

                    if self._WORD_DIMS==-1:
                        self._WORD_DIMS = len(values)-1

                    if len(values) == self._WORD_DIMS + 1:
                        coefs = np.asarray(values[1:], dtype="float32")
                        self._w2v[word] = coefs


    def get_train_word_vectorization(self, swap_questions, re_weight):
        #train_tokens1 = self._load_words('train_tokens1.pkl')
        # train_tokens2 = self._load_words('train_tokens2.pkl')
        train_tokens1 = self._load_words('train_lemmas1.pkl')
        train_tokens2 = self._load_words('train_lemmas2.pkl')
        y_train = np.load( os.path.join(self._data_folder, 'y_train.npy') )
        ntrain = len(train_tokens1)


        #self.submit_tokens1 = self._load_words('submit_tokens1.pkl')
        #self.submit_tokens2 = self._load_words('submit_tokens2.pkl')
        self.submit_tokens1 = self._load_words('submit_lemmas1.pkl')
        self.submit_tokens2 = self._load_words('submit_lemmas2.pkl')


        self._load_embeddings();

        self._token2freq = collections.Counter()
        self._max_words_len = 0
        for tokens in itertools.chain(train_tokens1, train_tokens2, self.submit_tokens1, self.submit_tokens2):
            slen = 0
            for token in tokens:
                if token in self._w2v:
                    self._token2freq[token] += 1
                    slen += 1
                self._max_words_len = max(self._max_words_len, slen)


        # оставим только TOP N самых частотных слов, которые есть в w2v модели
        TOP_TOKENS = 1000000
        good_tokens = set([token for (token, cnt) in sorted([(word, freq) for (word, freq) in self._token2freq.iteritems()], key=lambda z:-z[1])[:min(len(self._token2freq),TOP_TOKENS)]])

        self._token2index = {u'': 0}
        for (i, word) in enumerate(self._token2freq.keys()):
            if word != u'':
                self._token2index[word] = len(self._token2index)

        self._nb_words = len(self._token2index)

        nb_patterns = ntrain

        # Для инициализации матрицы весов в Embedding нам нужно для каждого
        # токена сделать вектор. Если токен присутствует в w2v модели, то вектор
        # берем оттуда, иначе делаем случайный.
        embedding_matrix = np.zeros((self._nb_words, self._WORD_DIMS))
        for (word,i) in self._token2index.iteritems():
            if word in self._w2v:
                embedding_matrix[i] = self._w2v[word]
            elif word != u'':
                embedding_matrix[i] = np.random.rand(self._WORD_DIMS)*2.0-1.0

        X1_data = np.zeros((nb_patterns, self._max_words_len), dtype=np.int32)
        X2_data = np.zeros((nb_patterns, self._max_words_len), dtype=np.int32)
        y_data = np.zeros(nb_patterns)

        n_ignored_tokens = 0  # для подсчета кол-ва слов, которые мы пропустим из-за их редкости
        # и отсутствия в w2v модели
        n_vectorized_tokens = 0
        idata = 0

        for i in range(ntrain):
            q1 = train_tokens1[i]
            q2 = train_tokens2[i]
            y = y_train[i]

            if self._ONLY_W2V_WORDS:
                q1 = [w for w in q1 if w in self._w2v]
                q2 = [w for w in q2 if w in self._w2v]

            y_data[idata] = bool(y)
            for (j, token) in enumerate(q1):
                if token == u'':
                    n_ignored_tokens += 1
                else:
                    if token in self._token2index:
                        X1_data[idata, j] = self._token2index[token]
                        n_vectorized_tokens += 1

            for (j, token) in enumerate(q2):
                if token == u'':
                    n_ignored_tokens += 1
                else:
                    if token in self._token2index:
                        X2_data[idata, j] = self._token2index[token]
                        n_vectorized_tokens += 1

            idata += 1

        #print('DEBUG: n_vectorized_tokens={} n_ignored_tokens={}'.format(n_vectorized_tokens,n_ignored_tokens))
        del train_tokens1
        del train_tokens2
        del self._w2v

        VALIDATION_SPLIT = 0.1
        perm = np.random.permutation(nb_patterns)
        idx_train = perm[:int(nb_patterns * (1 - VALIDATION_SPLIT))]
        idx_val = perm[int(nb_patterns * (1 - VALIDATION_SPLIT)):]

        if swap_questions:
            # удваиваем датасет, переставляя местами вопросы в парах
            X1_train = np.vstack((X1_data[idx_train], X2_data[idx_train]))
            X2_train = np.vstack((X2_data[idx_train], X1_data[idx_train]))
            y_train = np.concatenate((y_data[idx_train], y_data[idx_train]))

            X1_val = np.vstack((X1_data[idx_val], X2_data[idx_val]))
            X2_val = np.vstack((X2_data[idx_val], X1_data[idx_val]))
            y_val = np.concatenate((y_data[idx_val], y_data[idx_val]))
        else:
            X1_train =X1_data[idx_train]
            X2_train = X2_data[idx_train]
            y_train = y_data[idx_train]

            X1_val = X1_data[idx_val]
            X2_val = X2_data[idx_val]
            y_val = y_data[idx_val]

        weight_val = np.ones(len(y_val))
        if re_weight:
            weight_val *= 0.472001959
            weight_val[y_val == 0] = 1.309028344

        if re_weight:
            class_weight = {0: 1.309028344, 1: 0.472001959}
        else:
            class_weight = None

        WordTrainingDataset = collections.namedtuple('WordTrainingDataset',
                                                     ['X1_train', 'X2_train', 'y_train',
                                                      'X1_val', 'X2_val', 'y_val',
                                                      'weight_val', 'class_weight',
                                                      'nb_words', 'max_len',
                                                      'embedding_matrix'])

        return WordTrainingDataset(X1_train, X2_train, y_train,\
                                   X1_val, X2_val, y_val,\
                                   weight_val, class_weight,
                                   self._nb_words, self._max_words_len, embedding_matrix)


    def get_submit_word_vectorization(self):
        nb_submit = len(self.submit_tokens1)

        X1_submit = np.zeros((nb_submit, self._max_words_len), dtype=np.int32)
        X2_submit = np.zeros((nb_submit, self._max_words_len), dtype=np.int32)

        for idata, (tokens1, tokens2) in enumerate(zip(self.submit_tokens1, self.submit_tokens2)):

            if self._ONLY_W2V_WORDS:
                tx1 = [w for w in tokens1 if w in self._token2index]
                tx2 = [w for w in tokens2 if w in self._token2index]
            else:
                tx1 = tokens1
                tx2 = tokens2

            for (j, token) in enumerate(tx1):
                if token != u'' and token in self._token2index:
                    X1_submit[idata, j] = self._token2index[token]

            for (j, token) in enumerate(tx2):
                if token != u'' and token in self._token2index:
                    X2_submit[idata, j] = self._token2index[token]

        del self.submit_tokens1
        del self.submit_tokens2

        return (X1_submit, X2_submit)

    def store_stats(self):
        if len(self._token2freq)>0:
            with codecs.open(os.path.join(self._data_folder, 'token2freq.dat'), 'w', 'utf-8') as wrt:
                for word, cnt in sorted(self._token2freq.iteritems(), key=lambda z: -z[1]):
                    wrt.write(u'{}\t{}\n'.format(word, cnt))
