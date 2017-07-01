# -*- coding: utf-8 -*-
'''
Генерация word2vector моделей для слов.
Используется готовый корпус, в котором каждое слово отделено пробелами, и каждое
предложение находится на отдельной строке.
'''

from __future__ import print_function
from gensim.models import word2vec
import logging
import os
import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus_paths = [ 'training_corpus.txt',
                 os.path.expanduser('~/Corpus/word2vector/en/wiki.txt'),
               ]
SIZE=32
WINDOW=5
CBOW=1
MIN_COUNT=1


# ----------------------------------------------------------------------------

class MultiFilesReader:
    def __init__(self, fnames):
        self.fnames = fnames

    def __iter__(self):
        self.cur_file = 0
        self.reader = codecs.open(self.fnames[0], 'r', 'utf-8')
        return self;

    def next(self):
        line = self.reader.readline()
        if line == u'':
            self.reader.close()
            self.cur_file += 1
            if self.cur_file==len(self.fnames):
                raise StopIteration
            else:
                self.reader = codecs.open(self.fnames[self.cur_file], 'r', 'utf-8')
                line = self.reader.readline()
        else:
            line = line.strip()

        return line.split()


# ------------------------------------------------------------------------------

filename = 'w2v.CBOW=' + str(CBOW)+'_WIN=' + str(WINDOW) + '_DIM='+str(SIZE)

# в отдельный текстовый файл выведем все параметры модели
with open( filename + '.info', 'w+') as info_file:
    print('corpus_paths=', corpus_paths, file=info_file)
    print('SIZE=', SIZE, file=info_file)
    print('WINDOW=', WINDOW, file=info_file)
    print('CBOW=', CBOW, file=info_file)
    print('MIN_COUNT=', MIN_COUNT, file=info_file)

# начинаем обучение w2v
#sentences = word2vec.Text8Corpus(corpus_path)
#sentences = word2vec.LineSentence(corpus_path)


#for z in word2vec.LineSentence('training_corpus.txt'):
#    print(z)


sentences = MultiFilesReader(corpus_paths)

model = word2vec.Word2Vec(sentences, size=SIZE, window=WINDOW, cbow_mean=CBOW, min_count=MIN_COUNT, workers=4, sorted_vocab=1, iter=1 )
model.init_sims(replace=True)

# сохраняем готовую w2v модель
model.wv.save_word2vec_format( filename + '.model', binary=True)
#model.save_word2vec_format( filename + '.model.txt', binary=False)

