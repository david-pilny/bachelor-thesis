from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Training on pretrained model (30.10.2022):
#       rotation_range 40 -> 180
#       width_shift_range 0.2 -> 0.5
#       height_shift_range 0.2 -> 0.5
#       shear_range 0.2 -> 0.5
#       zoom_range 0.2 -> 0.7

training_output = '/storage/plzen1/home/david_pilny/Training_Output/'
training_id = '20230421'

model = load_model('./20221030_atherosclerosis_model.h5')

SIZE_X = 544
SIZE_Y = 544
n_classes = 4

input_dir = "./images/"
mask_dir = "./labels/"

train_images = []
for directory_path in sorted(os.listdir(input_dir)):
    input_path = input_dir + directory_path
    img = cv2.imread(input_path, 0)
    train_images.append(img)       
train_images = np.array(train_images)
print(train_images.shape)

train_masks = []
for directory_path in sorted(os.listdir(mask_dir)):
    input_path = mask_dir + directory_path
    mask = cv2.imread(input_path, 0)
    train_masks.append(mask)
train_masks = np.array(train_masks)
print(train_masks.shape)

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_reshaped_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_reshaped_encoded_original_shape)

train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks, axis=3)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state=42)

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

print("Class values in the dataset are ...", np.unique(y_train))

class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train_masks_reshaped_encoded),
                                        y = train_masks_reshaped_encoded)
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


checkpointer = ModelCheckpoint('{0}{1}_atherosclerosis_segmentation.h5'.format(training_output, training_id), verbose=1, save_best_only=True)
callbacks = [
    EarlyStopping(patience=2, monitor='val_loss'),
    TensorBoard(log_dir='logs')
]

batch_size = 16
seed = 100

img_data_gen_args = dict(
      rotation_range=180,
      width_shift_range=0.5,
      height_shift_range=0.5,
      shear_range=0.5,
      zoom_range=0.7,
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='reflect')

mask_data_gen_args = dict(
      rotation_range=180,
      width_shift_range=0.5,
      height_shift_range=0.5,
      shear_range=0.5,
      zoom_range=0.7,
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='reflect',
    )

image_data_generator = ImageDataGenerator(**img_data_gen_args)
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

image_data_generator.fit(X_train, augment=True, seed=seed)
image_generator = image_data_generator.flow(X_train, seed=seed)
valid_img_generator = image_data_generator.flow(X_test, seed=seed)


mask_data_generator.fit(y_train_cat, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train_cat, seed=seed)
valid_mask_generator = mask_data_generator.flow(y_test_cat, seed=seed)

train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)

steps_per_epoch = 3*(len(X_train))//batch_size


history = model.fit(train_generator, 
                    verbose=1,
                    validation_data=val_generator, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, 
                    epochs=15)

model.save('{0}{1}_atherosceloris_model.h5'.format(training_output, training_id))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(1, len(loss)+1)

plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss ')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('{0}{1}_training_loss.png'.format(training_output, training_id))
plt.clf()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs=range(1, len(loss)+1)

plt.plot(epochs, accuracy, 'y', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('{0}{1}_training_accuracy.png'.format(training_output, training_id))


for i in range(0, 10):
    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction=(model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    plt.figure(figsize=(9, 5))
    plt.subplot(231)
    plt.title('Testing image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing mask')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on testing image')
    plt.imshow(predicted_img, cmap='gray')
    plt.savefig('{0}{1}_test_prediction_{2}.png'.format(training_output, training_id, i))
