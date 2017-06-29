# -*- coding: utf-8 -*-
'''
Recurrent+convolutional char-level deep neural network для решения задачи конкурса
https://www.kaggle.com/c/quora-question-pairs
(c) Илья Козиев 2017 inkoziev@gmail.com  http://kelijah.livejournal.com
'''

from __future__ import print_function

import os
import codecs
import math
import gc
import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Lambda
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
import keras.optimizers
from QuestionPairsVectorizer import CharVectorizer
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM, GRU



CHAR_DIMS = 32
DROPOUT_RATE = 0.05
RE_WEIGHT = True
RECUR_SIZE = 2*CHAR_DIMS

DO_TRAIN = True
MAKE_SUBMISSION = True
DOUBLE_DATASET = False


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

# посмотрим, какие слова встретились в вопросах.
vectorizer.store_stats();

print('nb_words={}'.format(nb_chars))
print('max number of chars per line={}'.format(max_sent_len))

gc.collect()

# --------------------------------------------------------

# <editor-fold desc="Создание и обучение модели">

# разделяемый слой для векторизации символов
embedding_layer = Embedding(output_dim=CHAR_DIMS,
                            input_dim=nb_chars,
                            input_length=max_sent_len,
                            mask_zero=False,
                            trainable=True)


rnn_layer = Bidirectional( LSTM(RECUR_SIZE,
                                return_sequences=False,
                                dropout=DROPOUT_RATE,
                                recurrent_dropout=DROPOUT_RATE) )
rnn_out_size = RECUR_SIZE*2

#rnn_layer = GRU(RECUR_SIZE,
#                                return_sequences=False,
#                                dropout=DROPOUT_RATE,
#                                recurrent_dropout=DROPOUT_RATE)
#rnn_out_size = RECUR_SIZE


# сверточные слои с разным размером окна

nb_filters = CHAR_DIMS

conv_layer1 = Conv1D(filters=nb_filters,
                kernel_size=3,
                padding='valid',
                activation='relu',
                strides=1)

conv_layer2 = Conv1D(filters=nb_filters,
                kernel_size=4,
                padding='valid',
                activation='relu',
                strides=1)

conv_layer3 = Conv1D(filters=nb_filters,
                kernel_size=5,
                padding='valid',
                activation='relu',
                strides=1)

conv_layer4 = Conv1D(filters=nb_filters,
                kernel_size=6,
                padding='valid',
                activation='relu',
                strides=1)

conv_layer5 = Conv1D(filters=nb_filters,
                kernel_size=7,
                padding='valid',
                activation='relu',
                strides=1)

conv_layer6 = Conv1D(filters=nb_filters,
                kernel_size=8,
                padding='valid',
                activation='relu',
                strides=1)


# -----------------------------------

left_input = Input(shape=(max_sent_len,), dtype='int16')
right_input = Input(shape=(max_sent_len,), dtype='int16')

left_emb = embedding_layer(left_input)
right_emb = embedding_layer(right_input)

left_seq1 = conv_layer1(left_emb)
left_seq1 = GlobalMaxPooling1D()(left_seq1)

left_seq2 = conv_layer2(left_emb)
left_seq2 = GlobalMaxPooling1D()(left_seq2)

left_seq3 = conv_layer3(left_emb)
left_seq3 = GlobalMaxPooling1D()(left_seq3)

left_seq4 = conv_layer4(left_emb)
left_seq4 = GlobalMaxPooling1D()(left_seq4)

left_seq5 = conv_layer5(left_emb)
left_seq5 = GlobalMaxPooling1D()(left_seq5)

left_seq6 = conv_layer6(left_emb)
left_seq6 = GlobalMaxPooling1D()(left_seq6)

left_seq7 = rnn_layer(left_emb)


right_seq1 = conv_layer1(right_emb)
right_seq1 = GlobalMaxPooling1D()(right_seq1)

right_seq2 = conv_layer2(right_emb)
right_seq2 = GlobalMaxPooling1D()(right_seq2)

right_seq3 = conv_layer3(right_emb)
right_seq3 = GlobalMaxPooling1D()(right_seq3)

right_seq4 = conv_layer4(right_emb)
right_seq4 = GlobalMaxPooling1D()(right_seq4)

right_seq5 = conv_layer5(right_emb)
right_seq5 = GlobalMaxPooling1D()(right_seq5)

right_seq6 = conv_layer6(right_emb)
right_seq6 = GlobalMaxPooling1D()(right_seq6)

right_seq7 = rnn_layer(right_emb)

merged_size = nb_filters*6+rnn_out_size

# -----------------------------------

merged_left = keras.layers.concatenate([left_seq1, left_seq2, left_seq3, left_seq4, left_seq5, left_seq6, left_seq7])
merged_right = keras.layers.concatenate([right_seq1, right_seq2, right_seq3, right_seq4, right_seq5, right_seq6, right_seq7])

diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(merged_size,))([merged_left, merged_right])
mul = Lambda(lambda x: x[0] * x[1], output_shape=(merged_size,))([merged_left, merged_right])
merged = keras.layers.concatenate([diff, mul])

#merged = BatchNormalization()(merged)

merged = Dense(CHAR_DIMS*4, activation='relu')(merged)
merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)

merged = Dense(CHAR_DIMS*2, activation='relu')(merged)
merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)

merged = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[left_input, right_input], outputs=merged)

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

#open('chars_siamese3.arch', 'w').write(model.to_json())

print(model.summary())

if DO_TRAIN:
    model_checkpoint = ModelCheckpoint('chars_siamese3.model', monitor='val_loss',
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

    history = model.fit(x=[X1_train, X2_train],\
                        y=y_train, \
                        validation_data=([X1_val, X2_val], y_val, weight_val),\
                        class_weight=class_weight,\
                        batch_size=1024,
                        epochs=100,
                        callbacks=[model_checkpoint, early_stopping])

    val_loss = np.array(history.history['val_loss'])
    np.savetxt(os.path.join(submission_folder, "val_loss_2.csv"), val_loss, delimiter="\t")

# </editor-fold>


# <editor-fold desc="Генерация сабмита с помощью обученной модели">
if MAKE_SUBMISSION:
    print('Vectorization of submission datasets...')
    (X1_submit, X2_submit) = vectorizer.get_submit_char_vectorization()

    print("Computing submission y's...")
    model.load_weights('chars_siamese3.model')
    y1 = model.predict([X1_submit, X2_submit])[:, 0]
    if DOUBLE_DATASET:
        y2 = model.predict([X2_submit, X1_submit])[:, 0]

    submission_filename = os.path.join(submission_folder, 'submission_chars_3.dat')
    print('Storing submission to {}'.format(submission_filename))

    if DOUBLE_DATASET:
        store_submission( map( math.sqrt, y1*y2 ), submission_filename)
    else:
        store_submission( y1, submission_filename)

# </editor-fold>
