import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.utils import to_categorical
from tensorflow.keras.layers import Lambda, LeakyReLU
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Multiply, Dot, Softmax
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

data = pd.read_csv(r"icml_face_data.csv")

def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))
    
    ''' Combine 'Angry' and 'Disgust' into a single 'Negative' category.
        Merge 'Surprise' and 'Fear' into 'Startled'.
        Keep 'Happy', 'Sad', and 'Neutral' as they are, resulting in a simplified model with 4 classes instead of 7.
    '''
    
    image_label[image_label==1]=0
    image_label[image_label==2]=1
    image_label[image_label==3]=2
    image_label[image_label==4]=3
    image_label[image_label==5]=1
    image_label[image_label==6]=4

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def resize_images(image_array, size=(224, 224)):
    resized_images = np.zeros((len(image_array), size[0], size[1]))
    for i, img in enumerate(image_array):
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        resized_images[i] = img_resized

    return resized_images


def random_erasing(img, probability=0.3, sl=0.1, sh=0.2, r1=1):
    '''
    parameters:
    - img: original image
    - probability: the probability to conduct radom erasing
    - sl, sh: Erase the minimum and maximum ranges of the proportion of the area to the image area.
    - r1: Erase the area's aspect ratio range.
    '''
    if np.random.rand() > probability:
        return img  # return original image directly

    h, w, _ = img.shape
    area = h * w

    erase_area = np.random.uniform(sl, sh) * area
    aspect_ratio = np.random.uniform(r1, 1 / r1)

    h_erase = int(np.sqrt(erase_area / aspect_ratio))
    w_erase = int(np.sqrt(erase_area * aspect_ratio))

    if h_erase >= h or w_erase >= w:
        return img

    x = np.random.randint(0, h - h_erase)
    y = np.random.randint(0, w - w_erase)

    img[x:x + h_erase, y:y + w_erase, :] = 0
    return img


# function to print the distribution
def plot_compare_distributions(array1, array2, array3, title1='', title2='', title3=''):
    df_array1 = pd.DataFrame()
    df_array2 = pd.DataFrame()
    df_array3 = pd.DataFrame()
    df_array1['emotion'] = array1.argmax(axis=1)
    df_array2['emotion'] = array2.argmax(axis=1)
    df_array3['emotion'] = array3.argmax(axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=False)
    x = emotions.values()

    y = df_array1['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    axs[0].bar(x, y.sort_index(), color='orange')
    axs[0].set_title(title1)
    axs[0].grid()

    y = df_array2['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    axs[1].bar(x, y.sort_index())
    axs[1].set_title(title2)
    axs[1].grid()

    y = df_array3['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    axs[2].bar(x, y.sort_index(), color='black')
    axs[2].set_title(title3)
    axs[2].grid()

    plt.savefig('distribution.png')


emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
train_image_array, train_image_label = prepare_data(data[data[' Usage'] == 'Training'])
val_image_array, val_image_label = prepare_data(data[data[' Usage'] == 'PrivateTest'])
test_image_array, test_image_label = prepare_data(data[data[' Usage'] == 'PublicTest'])

train_images = resize_images(train_image_array).reshape((train_image_array.shape[0], 224, 224, 1))
train_images = train_images.astype('float32') / 255

val_images = resize_images(val_image_array).reshape((val_image_array.shape[0], 224, 224, 1))
val_images = val_images.astype('float32') / 255

test_images = resize_images(test_image_array).reshape((test_image_array.shape[0], 224, 224, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_image_label)
val_labels = to_categorical(val_image_label)
test_labels = to_categorical(test_image_label)

'''show the distribution'''
plot_compare_distributions(train_labels, val_labels,test_labels, title1='train labels', title2='val labels',title3='test labels')


'''define the model with attention'''
def add_self_attention_block(feature_map):
    # acquire feature map dynamically
    shape = tf.shape(feature_map)
    channels = feature_map.shape[-1]

    feature_map_reshaped = tf.reshape(feature_map, [shape[0], shape[1] * shape[2], channels])

    # define Q, K, V weights
    query = Dense(units=channels)(feature_map_reshaped)
    key = Dense(units=channels)(feature_map_reshaped)
    value = Dense(units=channels)(feature_map_reshaped)

    # calculate attention scores
    attention_scores = Dot(axes=-1)([query, key])
    attention_distribution = Softmax(axis=-1)(attention_scores)

    # apply attention weights
    attention_output = Dot(axes=1)([attention_distribution, value])

    attention_output_reshaped = tf.reshape(attention_output, [shape[0], shape[1], shape[2], channels])

    return attention_output_reshaped

def build_model_with_attention(input_shape=(224, 224, 1), num_classes=7):

    inputs = Input(shape=input_shape)

    # change the channels
    x = Lambda(lambda x: tf.tile(x, multiples=[1, 1, 1, 3]))(inputs)

    # download compare model
    base_model = VGG19(weights='imagenet', include_top=False, input_tensor=x)

    # don't use the final layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # add attention block
    attention_feature_map = add_self_attention_block(base_model.output)

    # use leakyrelu
    x = Flatten()(attention_feature_map)
    x = Dense(128, kernel_initializer='uniform')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(128, kernel_initializer='uniform')(x)
    x = LeakyReLU(alpha=0.1)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


model= build_model_with_attention()
model.summary()
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=[0.8, 1.2],
    fill_mode='reflect',
    brightness_range=[0.8,1.2],
    preprocessing_function=random_erasing()
)

train_generator = train_datagen.flow(
    train_images,
    train_labels,
    batch_size=64
)

val_datagen = ImageDataGenerator()

val_generator = val_datagen.flow(
    val_images,
    val_labels,
    batch_size=64
)

checkpoint_path = "merge_best_model.h5" 
model_checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True,
    mode='min' 
)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[model_checkpoint, reduce_lr] ) 

model = load_model("best_model.h5")

#evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test caccuracy:', test_acc)
pred_test_labels = model.predict(test_images)


'''analyse coverage'''
#loss
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.subplot(1,2,1)
plt.plot(epochs, loss, 'bo', label='loss_train')
plt.plot(epochs, loss_val, 'b', label='loss_val')
plt.title('value of the loss function')
plt.xlabel('epochs')
plt.ylabel('value of the loss function')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('merge_training_validation_loss.png')

#acc
acc = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(1, len(loss)+1)
plt.subplot(1,2,2)
plt.plot(epochs, acc, 'bo', label='accuracy_train')
plt.plot(epochs, acc_val, 'b', label='accuracy_val')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('value of accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('merge_training_validation_acc.png')


#confusion matrix
df_compare = pd.DataFrame()
df_compare['real'] = test_labels.argmax(axis=1)
df_compare['pred'] = pred_test_labels.argmax(axis=1)
df_compare['wrong'] = np.where(df_compare['real']!=df_compare['pred'], 1, 0)
conf_mat = confusion_matrix(test_labels.argmax(axis=1), pred_test_labels.argmax(axis=1))

fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                show_normed=True,
                                show_absolute=False,
                                class_names=emotions.values(),
                                figsize=(8, 8))
fig.savefig('merge_confusion_matrix.png')