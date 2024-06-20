# ## LSTM + Deep CNN Text Classification with Keras
# 

get_ipython().run_line_magic('run', 'Setup.ipynb')


# Load embedding matrix into an `Embedding` layer. Toggle `trainable=False` to prevent the weights from being updated during training.
# 

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# ### LSTM + Deep CNN Structure
# [Reference](https://github.com/richliao/textClassifier), [LTSM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), [CNN Source](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) and [Notes](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
# 
# Deeper CNN as described in [CNN for Sentence Classification](http://www.aclweb.org/anthology/D14-1181) (Yoon Kim, 2014), multiple filters have been applied. This can be implemented using Keras `Merge` Layer.
# 

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm1 = Bidirectional(LSTM(4,dropout=0.3,recurrent_dropout=0.3,return_sequences=True))(embedded_sequences)

convs, filter_sizes = [], [3,4,5]
for fsz in filter_sizes:
    l_conv = Conv1D(filters=32,kernel_size=fsz,
                    activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_lstm1)
    convs.append(l_conv)

l_merge = Concatenate(axis=1)(convs)
l_pool1 = MaxPooling1D(2)(l_merge)
l_drop1 = Dropout(0.4)(l_pool1)
l_flat = Flatten()(l_drop1)
l_dense = Dense(16, activation='relu')(l_flat)

preds = Dense(4, activation='softmax')(l_dense)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
lr_metric = get_lr_metric(adadelta)
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc', lr_metric])


def step_cyclic(epoch):
    try:
        l_r, decay = 1.0, 0.0001
        if epoch%33==0:multiplier = 10
        else:multiplier = 1
        rate = float(multiplier * l_r * 1/(1 + decay * epoch))
        #print("Epoch",epoch+1,"- learning_rate",rate)
        return rate
    except Exception as e:
        print("Error in lr_schedule:",str(e))
        return float(1.0)
    
def initial_boost(epoch):
    if epoch==0: return float(6.0)
    else: return float(1.0)
        
tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=4, batch_size=16, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("checkpoint.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_schedule = callbacks.LearningRateScheduler(step_cyclic)


model.summary()
print("Training Progress:")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=200, batch_size=50,
          callbacks=[tensorboard, model_checkpoints, lr_schedule])


model.save('ltsm-c.h5')





# ## CNN Text Classification with Keras
# 

import plaidml.keras
plaidml.keras.install_backend()

get_ipython().run_line_magic('run', 'Setup.ipynb')
get_ipython().run_line_magic('run', 'ExtraFunctions.ipynb')


# Load embedding matrix into an `Embedding` layer. Toggle `trainable=False` to prevent the weights from being updated during training.
# 

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# ### Deeper 1D CNN
# [Reference](https://github.com/richliao/textClassifier), [Source](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) and [Notes](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
# 
# Deeper Convolutional neural network: In [CNN for Sentence Classification](http://www.aclweb.org/anthology/D14-1181) (Yoon Kim, 2014), multiple filters have been applied. This can be implemented using Keras `Merge` Layer.
# 

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

convs, filter_sizes = [], [2,3,5]
for fsz in filter_sizes:
    l_conv = Conv1D(filters=16,kernel_size=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(2)(l_conv)
    convs.append(l_pool)

l_merge = Concatenate(axis=1)(convs)
l_cov1= Conv1D(24, 3, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(2)(l_cov1)
l_drop1 = Dropout(0.4)(l_pool1)
l_cov2 = Conv1D(16, 3, activation='relu')(l_drop1)
l_pool2 = MaxPooling1D(17)(l_cov2) # global max pooling
l_flat = Flatten()(l_pool2)
l_dense = Dense(16, activation='relu')(l_flat)
preds = Dense(6, activation='softmax')(l_dense)


model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])


#tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("checkpoints", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_schedule = callbacks.LearningRateScheduler(initial_boost)


get_ipython().system('rm -R logs')


model.summary()
model.save("cnn_kim.h5")


print("Training Progress:")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=50)





# ## LSTM-Inception Text Classification with Keras
# 

get_ipython().run_line_magic('run', 'Setup.ipynb')
get_ipython().run_line_magic('run', 'ExtraFunctions.ipynb')


# Load embedding matrix into an `Embedding` layer. Toggle `trainable=False` to prevent the weights from being updated during training.
# 

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


# ### LSTM + Inception Structure
# [Reference](https://github.com/richliao/textClassifier), [LTSM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), [CNN Source](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) and [Notes](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
# 
# Deeper CNN as described in [CNN for Sentence Classification](http://www.aclweb.org/anthology/D14-1181) (Yoon Kim, 2014), multiple filters have been applied. This can be implemented using Keras `Merge` Layer.
# 

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm1 = Bidirectional(LSTM(4,dropout=0.3,recurrent_dropout=0.3,return_sequences=True))(embedded_sequences)

# inception module
inception, filter_sizes = [], [3,4,6,7]
for fsz in filter_sizes:
    l_conv_i1 = Conv1D(filters=12,kernel_size=1,
                    activation='relu',)(l_lstm1)
    l_conv_i2 = Conv1D(filters=12,kernel_size=fsz,
                       activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_conv_i1)
    inception.append(l_conv_i2)
l_pool_inc = MaxPooling1D(4)(l_lstm1)
l_conv_inc = Conv1D(filters=12,kernel_size=1,
                    activation='relu',kernel_regularizer=regularizers.l2(0.02))(l_pool_inc)
inception.append(l_conv_inc)

l_merge = Concatenate(axis=1)(inception)
l_pool1 = MaxPooling1D(3)(l_merge)
l_drop1 = Dropout(0.4)(l_pool1)
l_flat = Flatten()(l_drop1)
l_dense = Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.02))(l_flat)

preds = Dense(4, activation='softmax')(l_dense)


model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
lr_metric = get_lr_metric(adadelta)
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc', lr_metric])


tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("checkpoint.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_schedule = callbacks.LearningRateScheduler(initial_boost)


model.summary()
model.save('ltsm-inception.h5')
print("Training Progress:")
model_log = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=200, batch_size=50,
          callbacks=[tensorboard, lr_schedule])


model.save('ltsm-inception.h5')


import pandas as pd
pd.DataFrame(model_log.history).to_csv("history-inception.csv")


