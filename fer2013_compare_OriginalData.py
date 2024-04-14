import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.utils import to_categorical
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, VGG16,ResNet50,DenseNet121
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dense, GlobalAveragePooling2D, Reshape, Multiply, Add, Flatten,Lambda
from keras.models import load_model


data = pd.read_csv(r"icml_face_data.csv")

def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label

def plot_compare_distributions(array1, array2,array3, title1='', title2='',title3=''):
    df_array1 = pd.DataFrame()
    df_array2 = pd.DataFrame()
    df_array3 = pd.DataFrame()
    df_array1['emotion'] = array1.argmax(axis=1)
    df_array2['emotion'] = array2.argmax(axis=1)
    df_array3['emotion'] = array3.argmax(axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
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
    
    plt.savefig('vgg_original_fer2013_distribution.png')

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
train_image_array, train_image_label = prepare_data(data[data[' Usage']=='Training'])
val_image_array, val_image_label = prepare_data(data[data[' Usage']=='PrivateTest'])
test_image_array, test_image_label = prepare_data(data[data[' Usage']=='PublicTest'])

train_images =train_image_array.reshape((train_image_array.shape[0], 48, 48, 1))
train_images = train_images.astype('float32')/255
val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))
val_images = val_images.astype('float32')/255
test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))
test_images = test_images.astype('float32')/255
train_labels = to_categorical(train_image_label)
val_labels = to_categorical(val_image_label)
test_labels = to_categorical(test_image_label)

'''show the distribution'''
plot_compare_distributions(train_labels, val_labels,test_labels, title1='train labels', title2='val labels',title3='test labels')

'''define the compare model'''
def build_compare_model(input_shape=(224, 224, 1), num_classes=7):
    input_tensor = Input(shape=input_shape)

    # change the channels
    x = Lambda(lambda x: tf.repeat(x, 3, axis=-1))(input_tensor)

    base_model = VGG19(weights="imagenet", include_top=False, input_tensor=x)
    # base_model = VGG19(weights=None, include_top=False, input_tensor=x)
    # base_model = ResNet50(weights=None, include_top=False, input_tensor=x)
    # base_model = DenseNet121(weights=None, include_top=False, input_tensor=x)

    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    output_tensor = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

model = build_compare_model()
model.summary()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator() 
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



checkpoint_path = "best_model.h5"
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss', 
    verbose=1,
    save_best_only=True,
    mode='min' 
)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto')


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[model_checkpoint, reduce_lr]
)





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
plt.savefig('vgg_original_fer2013_training_validation_loss.png')

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
plt.savefig('vgg_original_fer2013_training_validation_acc.png')


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
fig.savefig('vgg_original_fer2013_confusion_matrix.png')