import csv
import numpy as np
import cv2

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from keras.utils.visualize_util import plot

tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, ELU, Lambda, BatchNormalization
from keras.optimizers import Adam

import matplotlib.pyplot as plt

col_size, row_size = 64, 16
data_folder = "combined_data/"
csv_path = data_folder + "driving_log.csv"

def read_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)    # Skip headers
        return [r for r in reader]

def get_image(path):
    raw_image = cv2.imread(path)
    return process_image(raw_image)


def get_training_data():
    data = read_csv(csv_path)

    data_len = len(data)
    num_features = data_len * 3   # left, right, center image per data row

    X_train = np.empty((num_features, row_size, col_size, 3), dtype=np.float32)
    y_train = np.empty(num_features)

    for i in tqdm(range(data_len)):
        row = data[i]
        steer_angle = float(row[3])

        # Get Images
        center_image = get_image(data_folder + row[0].strip())
        left_image = get_image(data_folder + row[1].strip())
        right_image = get_image(data_folder + row[2].strip())

        # Get feature indices
        center_index = i*3
        left_index = i*3 + 1
        right_index = i*3 + 2

        # Set features
        X_train[center_index] = center_image
        X_train[left_index] = left_image
        X_train[right_index] = right_image

        # Set labels
        y_train[center_index] = steer_angle
        y_train[left_index] = steer_angle + 0.25     # Add 0.25 to steer_angle to drive back toward center
        y_train[right_index] = steer_angle - 0.25    # Subtract 0.25 from steer angle to drive back toward centeread_csv

    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train

def process_image(image):
    # Removing the top 55 pixels of the images to remove the sky.
    # Also removing the bottom 25 pixels to remove the dash of the car.
    # Convert to YUV
    # Resize to 16*64
    cropped_image = image[55:135, :, :]
    yuv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2YUV)
    resized_image = cv2.resize(yuv_image, (col_size, row_size), interpolation=cv2.INTER_AREA)
    return resized_image

# Not used
def augment_shift(image, steer_angle):
    x_shift_range = 24
    y_shift_range = 4
    x_shift = x_shift_range * np.random.uniform(-0.5,0.5)
    y_shift = y_shift_range * np.random.uniform(-0.5,0.5)
    M = np.float32([[1,0,x_shift],[0,1,y_shift]])
    new_steer_angle = steer_angle + x_shift*0.03
    shift_image = cv2.warpAffine(image,M,(col_size,row_size))
    return shift_image, new_steer_angle

def augment_brightness(image):
    brightness_adjuster = 0.25 + np.random.uniform()
    image[:,:,0] = image[:,:,0] * brightness_adjuster
    return image

def augment_image(image, steer_angle):
    preprocessed_image = augment_brightness(image)
    # preprocessed_image, steering_angle = augment_shift(preprocessed_image, steer_angle)
    if np.random.choice([True, False]):
        preprocessed_image = cv2.flip(image, 1) # Flip images left/right (over vertical axis)
        steer_angle *= -1
    return preprocessed_image, steer_angle

def training_batch_generator(X_train, y_train, batch_size):
    while True:
        X_batch = np.zeros((batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        y_batch = np.zeros(batch_size)
        for batch_index in range(batch_size):
            keep = False
            while not keep:
                rand_num = np.random.randint(len(X_train))
                image = X_train[rand_num]
                steer_angle = y_train[rand_num]
                x, y = augment_image(image, steer_angle)
                if abs(y) < 0.1:
                    if np.random.uniform() > 0.3:
                        keep = True
                # elif abs(y) > 0.7:
                #     keep = False
                else:
                    keep = True
            X_batch[batch_index] = x
            y_batch[batch_index] = y
        yield X_batch, y_batch

def val_generator(X_train, y_train):
    while True:
        rand_num = np.random.randint(len(X_train))
        image = X_train[rand_num]
        steer_angle = y_train[rand_num]
        x, y = augment_image(image, steer_angle)
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        y = np.array([[y]])
        yield x, y

def get_model():
    return Sequential([Lambda(lambda x: x/255.-0.5,input_shape=(row_size,col_size,3)),
                        Convolution2D(32, 3, 3, border_mode='same', init='he_normal', name='CNN_1_1'),
                        ELU(),
                        Convolution2D(32, 3, 3, border_mode='same', init='he_normal', name='CNN_1_2'),
                        ELU(),
                        MaxPooling2D(name='Max_Pooling_1'),
                        Dropout(0.5),
                        Convolution2D(64, 3, 3, border_mode='same', init='he_normal', name='CNN_2_1'),
                        ELU(),
                        Convolution2D(64, 3, 3, border_mode='same', init='he_normal', name='CNN_2_2'),
                        ELU(),
                        MaxPooling2D(name='Max_Pooling_2'),
                        Dropout(0.5),
                        Convolution2D(128, 3, 3, border_mode='same', init='he_normal', name='CNN_3_1'),
                        ELU(),
                        Convolution2D(128, 3, 3, border_mode='same', init='he_normal', name='CNN_3_2'),
                        ELU(),
                        MaxPooling2D(name='Max_Pooling_3'),
                        Dropout(0.5),
                        Flatten(),
                        Dense(512, init='he_normal', name='Dense_512'),
                        ELU(),
                        Dropout(0.5),
                        Dense(64, init='he_normal', name='Dense_64'),
                        ELU(),
                        Dropout(0.5),
                        Dense(1, name='output')])


if __name__ == "__main__":
    model = get_model()
    model.compile(optimizer=Adam(0.0001), loss='mse')

    # plot(model, to_file='model.png')

    X_train, y_train = get_training_data()
    batch_size = 256
    batch_generator = training_batch_generator(X_train, y_train, batch_size)

    plt.hist(y_train, bins=50)
    plt.show()

    val_size = 5000
    val_gen = val_generator(X_train, y_train)

    model_checkpoint = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', verbose=1,
                                       save_best_only=False, save_weights_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

    model.fit_generator(batch_generator, samples_per_epoch=12800, nb_epoch=100, verbose=1,
                        validation_data=val_gen, nb_val_samples=val_size,
                        callbacks=[model_checkpoint, early_stop])

    model.save_weights('model.h5')
    json = model.to_json()
    with open('model.json', 'w') as out:
        out.write(json)