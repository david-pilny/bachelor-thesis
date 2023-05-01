import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_output_dir(arr):
    root_path = os.path.join(arr[0], arr[1])
    arr = arr[1:]
    arr[0] = root_path
    if os.path.exists(root_path) == False:
        print('Creating new directory ...')
        os.mkdir(root_path)
    if len(arr) > 1:
        create_output_dir(arr)
        
def create_dataset(input_dir, 
                   mask_dir, 
                   output_dir_images, 
                   output_dir_masks, 
                   img_data_gen_args, 
                   mask_data_gen_args):
    SIZE_X = 544
    SIZE_Y = 544
    n_classes = 4
    batch_size = 16
    seed = 100
    
    if os.path.exists(input_dir) == False:
        print('Input directory does not exist !!!')
        return
    
    if os.path.exists(mask_dir) == False:
        print('Mask directory does not exist !!!')
        return
    
    create_output_dir((os.path.normpath(output_dir_images)).split(os.sep))
    create_output_dir((os.path.normpath(output_dir_masks)).split(os.sep))
    
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
    
    train_images = np.expand_dims(train_images, axis=3)
    train_masks_input = np.expand_dims(train_masks, axis=3)
    
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    
    image_data_generator.fit(train_images, augment=True, seed=seed)
    image_generator = image_data_generator.flow(train_images, seed=seed)

    mask_data_generator.fit(train_masks_input, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(train_masks_input, seed=seed)
    
    img_generator = zip(image_generator, mask_generator)
    
    generated_images = []
    generated_labels = []
    index = 0

    for (img, mask) in img_generator:
        for i in img:
            generated_images.append(i)
        for m in mask:
            generated_labels.append(m)
        if index >= 32:
            break
        index += 1
        
    generated_images = np.array(generated_images)
    generated_labels = np.array(generated_labels)
    
    generated_images = generated_images.reshape(generated_images.shape[0],
                        generated_images.shape[1],
                        generated_images.shape[2])

    generated_labels = generated_labels.reshape(generated_labels.shape[0],
                        generated_labels.shape[1],
                        generated_labels.shape[2])
    index = 0
    for img in generated_images:
        cv2.imwrite(os.path.join(output_dir_images, '{0}.png'.format(index)), img)
        index += 1

    index = 0
    for mask in generated_labels:
        cv2.imwrite(os.path.join(output_dir_masks, '{0}.png'.format(index)), img)
        index += 1
    print('Dataset generated !')