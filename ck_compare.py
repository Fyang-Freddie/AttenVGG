import os
import numpy as np
import cv2
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
from tensorflow.keras.applications import VGG19,ResNet50,DenseNet121,VGG16
from tensorflow.keras.layers import Lambda
from keras.models import Model

'''define the model used for comparasion'''
def compare_model(input_shape=(224, 224, 1), num_classes=7):
    input_shape = (224, 224, 1)  # original input size
    num_classes = 7


    # define the input layers and adjust the channels
    inputs = Input(shape=input_shape)
    x = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)


    # download the pretrained model
    base_model = VGG16(weights=None, include_top=False, input_tensor=x)
    # base_model = VGG19(weights=None, include_top=False, input_tensor=x)
    # base_model = ResNet50(weights=None, include_top=False, input_tensor=x)
    # base_model = DenseNet121(weights=None, include_top=False, input_tensor=x)

    # add fully connect layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=predictions)

# Define path to the dataset
dataset_path = 'CK'

# Define emotions
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt']  # Adjust as per your folder names

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

'''ten folds method'''
# define KFold
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=16)

# save the acc of each epoch
acc_per_fold = []
loss_per_fold = []

# KFolds
fold_no = 1
for train, test in kf.split(data, labels):

    model=compare_model()
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f'Training for fold {fold_no} ...')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto') 
    # Create data generators for augmentation
    train_datagen = ImageDataGenerator(
    )
    val_datagen = ImageDataGenerator()

    # Configure batch size and retrieve data generator
    train_generator = train_datagen.flow(data[train], labels[train], batch_size=32)
    val_generator = val_datagen.flow(data[test],labels[test], batch_size=32)

    # training
    history = model.fit(train_generator,validation_data=val_generator,epochs=60,callbacks=reduce_lr)  
    scores = model.evaluate(data[test], labels[test])
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    # add fold
    fold_no += 1

# print the results
print(f'mean acc: {np.mean(acc_per_fold)}% +/- {np.std(acc_per_fold)}%')
print(f'mean loss: {np.mean(loss_per_fold)}')
