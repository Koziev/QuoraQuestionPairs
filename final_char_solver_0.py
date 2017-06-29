# -*- coding: utf-8 -*-
'''
Feedforward char-level deep neural network для решения задачи конкурса
https://www.kaggle.com/c/quora-question-pairs/data

(c) Илья Козиев 2017 inkoziev@gmail.com
'''

from __future__ import print_function

import os
import codecs
import math

from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape
from keras.models import Model
import keras.optimizers
from keras.layers import Lambda
from keras import backend as K
import keras.utils


from QuestionPairsVectorizer import CharVectorizer
from custom_loss import customized_loss1
import gc

CHAR_DIMS = 16
RE_WEIGHT = True
DROPOUT_RATE = 0.20

submission_folder = './submission'

# ------------------------------------------------------------------------

def store_submission(y_submission, submission_filename):
    with codecs.open(submission_filename, 'w') as wrt:
        wrt.write('test_id,is_duplicate\n')
        for idata, y in enumerate(y_submission):
            wrt.write('{},{}\n'.format(idata, y))

# -------------------------------------------------------------------------

vectorizer = CharVectorizer( max_chars_per_line=100 )

training_dataset = vectorizer.get_train_char_vectorization(swap_questions=True,re_weight=RE_WEIGHT)
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

# --------------------------------------------------------

# <editor-fold desc="Создание и обучение модели">

embedding_layer = Embedding(output_dim=CHAR_DIMS,
                            input_dim=nb_chars,
                            input_length=max_sent_len,
                            mask_zero=False,
                            trainable=True)

dense_units = 256
dense_units2 = 64
dense_layer = Dense(units=dense_units,activation='relu')
dense_layer2 = Dense(units=dense_units2,activation='relu')

left_input = Input( shape=(max_sent_len,), dtype='int16' )
left_seq = embedding_layer( left_input )
left_seq = Reshape((CHAR_DIMS*max_sent_len,))(left_seq)
left_seq = dense_layer(left_seq)
left_seq = Dropout(DROPOUT_RATE)(left_seq)

left_seq = BatchNormalization()(left_seq)
left_seq = dense_layer2(left_seq)
left_seq = Dropout(DROPOUT_RATE)(left_seq)


right_input = Input( shape=(max_sent_len,), dtype='int16' )
right_seq = embedding_layer( right_input )
right_seq = Reshape((CHAR_DIMS*max_sent_len,))(right_seq)
right_seq = dense_layer(right_seq)
right_seq = Dropout(DROPOUT_RATE)(right_seq)

right_seq = BatchNormalization()(right_seq)
right_seq = dense_layer2(right_seq)
right_seq = Dropout(DROPOUT_RATE)(right_seq)

merged = keras.layers.concatenate( [left_seq, right_seq] )

#diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(dense_units,))([left_seq, right_seq])
#mul = Lambda(lambda x: x[0] * x[1], output_shape=(dense_units,))([left_seq, right_seq])
#merged = keras.layers.concatenate([diff, mul])

merged = Dense(units=dense_units*2, activation='relu')(merged)
merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)

merged = Dense(units=dense_units, activation='relu')(merged)
merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)

merged = Dense(units=int(dense_units/2), activation='relu')(merged)
merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)

merged = Dense(units=int(dense_units/4), activation='relu')(merged)
merged = Dropout(DROPOUT_RATE)(merged)
merged = BatchNormalization()(merged)

merged = Dense(1, activation='sigmoid')(merged)
model = Model( inputs=[left_input, right_input], outputs=merged )


model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

#keras.utils.visualize_util.plot(model, to_file='final_char_solver_0.png', show_shapes=True)

#open('chars_siamese0.arch', 'w').write(model.to_json())

print(model.summary())

model_checkpoint = ModelCheckpoint('chars_siamese0.model', monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

history = model.fit(x=[X1_train, X2_train],\
                    y=y_train, \
                    validation_data=([X1_val, X2_val], y_val, weight_val),\
                    class_weight=class_weight,\
                    batch_size=256,\
                    epochs=100,\
                    callbacks=[model_checkpoint, early_stopping])

# </editor-fold>


# <editor-fold desc="Генерация сабмита с помощью обученной модели">
print('Vectorization of submission datasets...')
(X1_submit, X2_submit) = vectorizer.get_submit_char_vectorization()
gc.collect()

print("Computing submission y's...")
model.load_weights('chars_siamese0.model')
y1 = model.predict([X1_submit, X2_submit])[:, 0]
y2 = model.predict([X2_submit, X1_submit])[:, 0]
submission_filename = os.path.join(submission_folder, 'submission_chars_0.dat')
print('Storing submission to {}'.format(submission_filename))
store_submission( map( math.sqrt, y1*y2 ), submission_filename)
# </editor-fold>
