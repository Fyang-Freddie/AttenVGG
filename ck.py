import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dense, Reshape, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from keras.models import load_model


'''define the model with attention'''
def vgg_block(x, out_channels, conv_nums, if_maxpooling=True):
    for _ in range(conv_nums):
        x = Conv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    if if_maxpooling:
        x = MaxPooling2D(pool_size=2, strides=2)(x)
    return x
def Attention_block(input_shape=(224, 224, 1), num_classes=7):  # Adjusted for single-channel input
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = vgg_block(inputs, 64, conv_nums=1)
    x = vgg_block(x, 128, conv_nums=1)
    x = vgg_block(x, 256, conv_nums=2)
    x = vgg_block(x, 512, conv_nums=2)
    x = vgg_block(x, 512, conv_nums=2)
    
    # Reshape for self-attention
    reshaped_x = Reshape((7*7, 512))(x)  # Adjusted based on the output size of the last VGG block
    
    # Self-attention layers
    # Query, Key, Value transformations
    query = Dense(1)(reshaped_x)
    key = Dense(1)(reshaped_x)
    value = Dense(1)(reshaped_x)
    
    # Calculate attention scores
    scores = tf.matmul(query, key, transpose_b=True)
    distribution = tf.nn.softmax(scores)
    result = tf.matmul(distribution, value)
    
    # Reshape and concat
    result = Reshape((7*7,))(result)
    combined_features = Flatten()(x)
    combined_features = tf.keras.layers.Concatenate()([combined_features,result])
    
    # Fully connected layers
    x = Dense(4096, activation='relu')(combined_features)
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

model = Attention_block()
model.summary()


# Define path to the dataset
dataset_path = 'CK'

# Define emotions
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt']

# Initialize lists to store data and labels
data = []
labels = []

# Loop through all emotion folders
for i, emotion in enumerate(emotions):
    emotion_path = os.path.join(dataset_path, emotion)
    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if image is None:
            continue
        image = cv2.resize(image, (224, 224))  # Resize image to 224x224
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        data.append(image)
        labels.append(i)  # The index of the emotion in the emotions list is used as the label

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize pixel values
data = data / 255.0

# One-hot encode labels
labels = to_categorical(labels, num_classes=len(emotions))


#16:1,66:0.9898,26:0.98,36:0.9696,46:0.9797
# Initial split: 80% for training, 20% for temporary testing (which will be split again into validation and test)
X_train, X_temp_test, y_train, y_temp_test = train_test_split(data, labels, test_size=0.2, random_state=16)

# Split the temporary test set into validation and test sets (50% each of the temporary test set, resulting in 10% of the total dataset each)
X_val, X_test, y_val, y_test = train_test_split(X_temp_test, y_temp_test, test_size=0.5, random_state=16)


'''print the distribution'''
emotions_dict = {i: emotion for i, emotion in enumerate(emotions)}
def plot_compare_distributions(array1, array2, array3, title1='', title2='', title3=''):
    df_array1 = pd.DataFrame()
    df_array2 = pd.DataFrame()
    df_array3 = pd.DataFrame()
    df_array1['emotion'] = array1.argmax(axis=1)
    df_array2['emotion'] = array2.argmax(axis=1)
    df_array3['emotion'] = array3.argmax(axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    x = list(emotions_dict.values())

    for idx, (df, title) in enumerate(zip([df_array1, df_array2, df_array3], [title1, title2, title3])):
        y = df['emotion'].value_counts().reindex(range(len(emotions_dict)), fill_value=0)
        axs[idx].bar(x, y.sort_index(), color=['orange', 'blue', 'black'][idx])
        axs[idx].set_title(title)
        axs[idx].grid()

    plt.savefig('ck_distribution.png')
plot_compare_distributions(y_train, y_val, y_test, title1='Training Set', title2='Validation Set', title3='Test Set')


'''usual training method'''
# '''data augmentation'''
# # Create data generators for augmentation
# train_datagen = ImageDataGenerator(
#     rotation_range=20,
# )

# val_datagen = ImageDataGenerator()

# # Configure batch size and retrieve data generator
# train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
# val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

# checkpoint_path = "ck_best_model.h5"  #
# model_checkpoint = ModelCheckpoint( 
#     checkpoint_path, 
#     monitor='val_loss',  #
#     verbose=1, 
#     save_best_only=True,  # save the best model
#     mode='min'
# )

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto')

# # Pass both callbacks in a single list to the 'callbacks' parameter
# history = model.fit(train_generator,validation_data=val_generator,epochs=60, callbacks=[model_checkpoint, reduce_lr])

# model = load_model("ck_best_model.h5")
# #evaluate
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print('test caccuracy:', test_acc)
# pred_test_labels = model.predict(X_test)




'''ten folds method'''
# define KFold
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=16)

# save accuracy of each epoch
acc_per_fold = []
loss_per_fold = []

# KFold
fold_no = 1
for train, test in kf.split(data, labels):
    # define model
    model = Attention_block()

    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f'Training for fold {fold_no} ...')

    checkpoint_path = "ck_best_model.h5"
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',  verbose=1, save_best_only=True, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto')
    
    
    
    # Create data generators for augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
    )
    val_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(data[train], labels[train], batch_size=32)
    val_generator = val_datagen.flow(data[test],labels[test], batch_size=32)

    # train
    history = model.fit(train_generator,validation_data=val_generator,epochs=60,callbacks=[model_checkpoint, reduce_lr])
    model = load_model("ck_best_model.h5")    
    scores = model.evaluate(data[test], labels[test])
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    # add fold
    fold_no += 1

# print the result
print(f'mean accuracy: {np.mean(acc_per_fold)}% +/- {np.std(acc_per_fold)}%')
print(f'mean loss: {np.mean(loss_per_fold)}')





'''analyse coverage'''
# #loss
# loss = history.history['loss']
# loss_val = history.history['val_loss']
# epochs = range(1, len(loss)+1)
# plt.subplot(1,2,1)
# plt.plot(epochs, loss, 'bo', label='loss_train')
# plt.plot(epochs, loss_val, 'b', label='loss_val')
# plt.title('value of the loss function')
# plt.xlabel('epochs')
# plt.ylabel('value of the loss function')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('ck_training_validation_loss.png')

'''acc'''
# acc = history.history['accuracy']
# acc_val = history.history['val_accuracy']
# epochs = range(1, len(loss)+1)
# plt.subplot(1,2,2)
# plt.plot(epochs, acc, 'bo', label='accuracy_train')
# plt.plot(epochs, acc_val, 'b', label='accuracy_val')
# plt.title('accuracy')
# plt.xlabel('epochs')
# plt.ylabel('value of accuracy')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('ck_training_validation_acc.png')


'''confusion matrix'''
# df_compare = pd.DataFrame()
# df_compare['real'] = y_test.argmax(axis=1)
# df_compare['pred'] = pred_test_labels.argmax(axis=1)
# df_compare['wrong'] = np.where(df_compare['real']!=df_compare['pred'], 1, 0)
# conf_mat = confusion_matrix(y_test.argmax(axis=1), pred_test_labels.argmax(axis=1))

# fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
#                                 show_normed=True,
#                                 show_absolute=False,
#                                 class_names=emotions,
#                                 figsize=(8, 8))
# fig.savefig('ck_confusion_matrix.png')

