# -*- coding: utf-8 -*-
'''
Recurrent char-level deep neural network для решения задачи конкурса
https://www.kaggle.com/c/quora-question-pairs/data
(c) Илья Козиев 2017 inkoziev@gmail.com
'''

from __future__ import print_function

import os
import codecs
import gc
import numpy as np
import math

from keras.layers import Dense, Dropout, Input
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import keras.optimizers
from keras import backend as K
from keras.layers import Lambda

from QuestionPairsVectorizer import CharVectorizer
from custom_loss import customized_loss1


CHAR_DIMS = 32
RECUR_SIZE = CHAR_DIMS*2
DROPOUT_RATE = 0.05
RE_WEIGHT = True
DOUBLE_DATASET = False

DO_TRAIN = True
MAKE_SUBMISSION = True

submission_folder = './submission'

# ------------------------------------------------------------------------

def store_submission(y_submission, submission_filename):
    with codecs.open(submission_filename, 'w') as wrt:
        wrt.write('test_id,is_duplicate\n')
        for idata, y in enumerate(y_submission):
            wrt.write('{},{}\n'.format(idata, y))


# -------------------------------------------------------------------------

vectorizer = CharVectorizer()

training_dataset = vectorizer.get_train_char_vectorization(swap_questions=DOUBLE_DATASET,re_weight=RE_WEIGHT)
X1_train = training_dataset.X1_train
X2_train = training_dataset.X2_train
y_train = training_dataset.y_train
X1_val = training_dataset.X1_val
X2_val = training_dataset.X2_val
y_val = training_dataset.y_val
weight_val = training_dataset.weight_val
class_weight = training_dataset.class_weight

nb_chars = training_dataset.nb_chars
max_sent_len = training_dataset.max_len

print('nb_chars={}'.format(nb_chars))
print('max_sent_len={} chars'.format(max_sent_len))

gc.collect()

# посмотрим, какие символы встретились в вопросах.
vectorizer.store_stats();

# --------------------------------------------------------

#<editor-fold desc="Создание и обучение модели">

embedding_layer = Embedding( output_dim=CHAR_DIMS,
                             input_dim=nb_chars,
                             input_length=max_sent_len,
                             mask_zero=True,
                             trainable=True )

# rnn_layer1 = Bidirectional( LSTM(RECUR_SIZE,
#                                 return_sequences=False,
#                                 dropout=DROPOUT_RATE,
#                                 recurrent_dropout=DROPOUT_RATE) )
#
# rnn_layer2 = Bidirectional( LSTM(RECUR_SIZE,
#                                 return_sequences=False,
#                                 dropout=DROPOUT_RATE,
#                                 recurrent_dropout=DROPOUT_RATE) )
# rnn_out_size = RECUR_SIZE*2

rnn_layer = GRU(RECUR_SIZE,
                                return_sequences=False,
                                dropout=DROPOUT_RATE,
                                recurrent_dropout=DROPOUT_RATE)
rnn_out_size = RECUR_SIZE



left_input = Input( shape=(max_sent_len,), dtype='int32' )
left_seq = embedding_layer( left_input )
left_seq = rnn_layer(left_seq)

right_input = Input( shape=(max_sent_len,), dtype='int32' )
right_seq = embedding_layer( right_input )
right_seq = rnn_layer(right_seq)

# первый вариант объединения - просто сцепляем оба вектора
#merged = keras.layers.concatenate( [left_seq, right_seq] )

# второй вариант объединения
diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(rnn_out_size,))([left_seq, right_seq])
mul = Lambda(lambda x: x[0] * x[1], output_shape=(rnn_out_size,))([left_seq, right_seq])
merged = keras.layers.concatenate([diff, mul])


print('merged.shape={}'.format(merged._keras_shape))

merged = BatchNormalization()(merged)

merged = Dense(RECUR_SIZE*2)(merged)
merged = PReLU()(merged)
merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)

merged = Dense(RECUR_SIZE)(merged)
merged = PReLU()(merged)
merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)

merged = Dense(1, activation='sigmoid')(merged)

model = Model( inputs=[left_input, right_input], outputs=merged )

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

open( 'chars_siamese1.arch', 'w').write(model.to_json())

print(model.summary())

if DO_TRAIN:
    model_checkpoint = ModelCheckpoint('chars_siamese1.model', monitor='val_loss',
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

    history = model.fit(x=[X1_train, X2_train],\
                        y=y_train, \
                        validation_data=([X1_val, X2_val], y_val, weight_val),\
                        class_weight=class_weight,\
                        batch_size=1024,
                        epochs=100,
                        callbacks=[model_checkpoint, early_stopping])

# </editor-fold>


#<editor-fold desc="Генерация сабмита с помощью обученной модели">
if MAKE_SUBMISSION:
    print('Vectorization of submission datasets...')
    (X1_submit, X2_submit) = vectorizer.get_submit_char_vectorization()
    
    print("Computing submission y's...")
    model.load_weights('chars_siamese1.model')
    y1 = model.predict([X1_submit, X2_submit])[:, 0]
    if DOUBLE_DATASET:
        y2 = model.predict([X2_submit, X1_submit])[:, 0]

    submission_filename = os.path.join(submission_folder, 'submission_chars_1.dat')
    print('Storing submission to {}'.format(submission_filename))

    if DOUBLE_DATASET:
        store_submission( map( math.sqrt, y1*y2 ), submission_filename)
    else:
        store_submission( y1, submission_filename)
# </editor-fold>

