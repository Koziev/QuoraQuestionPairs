# -*- coding: utf-8 -*-
'''
Recurrent word-level deep neural network для решения задачи конкурса
https://www.kaggle.com/c/quora-question-pairs

(c) Илья Козиев 2017 inkoziev@gmail.com
'''

from __future__ import print_function

import os
import codecs

from keras.layers import Dense, Dropout, Input, Permute, Flatten, Reshape, RepeatVector
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.layers import Lambda
from keras import backend as K
import keras.layers
from keras.layers.merge import concatenate, add, multiply

from QuestionPairsVectorizer import WordVectorizer
from custom_loss import customized_loss1
import gc
import math

DROPOUT_RATE = 0.10
RE_WEIGHT = True
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

def attention_3d_block(inputs ):
    SINGLE_ATTENTION_VECTOR = False
    # inputs.shape = (batch_size, time_steps, input_dim)
    # input_dim = int(inputs.shape[2])
    input_dim = int(inputs._keras_shape[2])
    time_steps = int(inputs._keras_shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1) )(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)  # , name='attention_vec'
    output_attention_mul = keras.layers.multiply([inputs, a_probs]) # , name='attention_mul'
    return output_attention_mul

# -------------------------------------------------------------------------

vectorizer = WordVectorizer()

training_dataset = vectorizer.get_train_word_vectorization(swap_questions=False, re_weight=RE_WEIGHT)

X1_train = training_dataset.X1_train
X2_train = training_dataset.X2_train
y_train = training_dataset.y_train

X1_val = training_dataset.X1_val
X2_val = training_dataset.X2_val
y_val = training_dataset.y_val

weight_val = training_dataset.weight_val
class_weight = training_dataset.class_weight

nb_words = training_dataset.nb_words
max_words_len = training_dataset.max_len
embedding_matrix = training_dataset.embedding_matrix

word_dims = embedding_matrix.shape[1]
recur_size = 2*word_dims

# посмотрим, какие слова встретились в вопросах.
vectorizer.store_stats();

print('nb_words={}'.format(nb_words))
print('max number of words per line={}'.format(max_words_len))

gc.collect()

# --------------------------------------------------------

# <editor-fold desc="Создание и обучение модели">

embedding_layer = Embedding(output_dim=word_dims,
                            input_dim=nb_words,
                            input_length=max_words_len,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            trainable=False)

recur_layer = Bidirectional(
    LSTM(recur_size, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE))

left_input = Input(shape=(max_words_len,), dtype='int32')
left_seq = embedding_layer(left_input)
left_seq = recur_layer(left_seq)

right_input = Input(shape=(max_words_len,), dtype='int32')
right_seq = embedding_layer(right_input)
right_seq = recur_layer(right_seq)

# ---------------- attention -----------------------

att_compress = Dense(units=word_dims*2, activation='relu')

left_attention_mul = attention_3d_block(left_seq)
left_seq = Flatten()(left_attention_mul)
left_seq = att_compress(left_seq)

right_attention_mul = attention_3d_block(right_seq)
right_seq = Flatten()(right_attention_mul)
right_seq = att_compress(right_seq)

# ----------------- entailment ------------

#merged = keras.layers.concatenate([left_seq, right_seq])
#print('merged.shape={}'.format(merged._keras_shape))

branch_out_size = int(left_seq._keras_shape[1])
print('branch_out_size={}'.format(branch_out_size))

diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(branch_out_size,))([left_seq, right_seq])
mul = Lambda(lambda x: x[0] * x[1], output_shape=(branch_out_size,))([left_seq, right_seq])
merged = keras.layers.concatenate([diff, mul])

merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)
merged = Dense(recur_size * 2, activation='relu')(merged)

merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)
merged = Dense(recur_size, activation='relu')(merged)

merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)
merged = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[left_input, right_input], outputs=merged)

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

open('words_siamese1.arch', 'w').write(model.to_json())

if DO_TRAIN:
    model_checkpoint = ModelCheckpoint('words_siamese1.model', monitor='val_loss',
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

    history = model.fit(x=[X1_train, X2_train],
                        y=y_train,
                        validation_data=([X1_val, X2_val], y_val, weight_val),
                        class_weight=class_weight,
                        batch_size=128,
                        epochs=100,
                        callbacks=[model_checkpoint, early_stopping])

# </editor-fold>


# <editor-fold desc="Генерация сабмита с помощью обученной модели">
if MAKE_SUBMISSION:
    print('Vectorization of submission datasets...')
    (X1_submit, X2_submit) = vectorizer.get_submit_word_vectorization()

    print("Computing submission y's...")
    model.load_weights('words_siamese1.model')
    y1 = model.predict([X1_submit, X2_submit])[:, 0]

    submission_filename = os.path.join(submission_folder, 'submission_words_1.dat')
    print('Storing submission to {}'.format(submission_filename))
    store_submission( y1, submission_filename)
# </editor-fold>

