import pickle
import numpy as np
import cv2

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, ELU
from keras.optimizers import Adam


def open_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_training_data():
    # Bug in OS X w/ pickle makes saving data over 2GB fail.  Break into multiple chunks.
    X_train = np.vstack([open_data('data/image_data' + str(i) + '.pickle') for i in range(2)])
    y_train = open_data('data/label_data.pickle')
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train

def augment_brightness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_adjuster = 0.3 + np.random.uniform()
    hsv_image[:,:,2] = hsv_image[:,:,2] * brightness_adjuster
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

def augment_image(image, steer_angle):
    preprocessed_image = augment_brightness(image)
    if np.random.choice([True, False]):
        preprocessed_image = cv2.flip(image, 1) # Flip images left/right (over vertical axis)
        steer_angle *= -1
    return preprocessed_image, steer_angle

def training_batch_generator(X_train, y_train, batch_size):
    while True:
        X_batch = np.zeros((batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        y_batch = np.zeros(batch_size)
        for batch_index in range(batch_size):
            rand_num = np.random.randint(len(X_train))
            image = X_train[rand_num]
            steer_angle = y_train[rand_num]
            x, y = augment_image(image, steer_angle)
            X_batch[batch_index] = x
            y_batch[batch_index] = y
        yield X_batch, y_batch


def get_model():
    return Sequential([BatchNormalization(input_shape=(64,64,3)),
                        Convolution2D(3, 1, 1, border_mode='same'),
                        ELU(),
                        Convolution2D(32, 3, 3, border_mode='same'),
                        ELU(),
                        Convolution2D(32, 3, 3, border_mode='same'),
                        ELU(),
                        MaxPooling2D(),
                        Dropout(0.3),
                        Convolution2D(64, 3, 3, border_mode='same'),
                        ELU(),
                        Convolution2D(64, 3, 3, border_mode='same'),
                        ELU(),
                        MaxPooling2D(),
                        Dropout(0.3),
                        Convolution2D(128, 3, 3, border_mode='same'),
                        ELU(),
                        Convolution2D(128, 3, 3, border_mode='same'),
                        ELU(),
                        MaxPooling2D(),
                        Dropout(0.3),
                        Flatten(),
                        Dense(512),
                        ELU(),
                        Dense(64),
                        ELU(),
                        Dense(1)])


if __name__ == "__main__":
    # model = get_model()
    # model.compile(optimizer=Adam(0.00001), loss='mean_squared_error')

    model_filename = "model.json"
    with open(model_filename, 'r') as jfile:
        # instead.
        model = model_from_json(jfile.read())
    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
    weights_file = model_filename.replace('json', 'h5')
    model.load_weights(weights_file)

    X_train, y_train = get_training_data()
    batch_size = 256
    batch_generator = training_batch_generator(X_train, y_train, batch_size)

    model.fit_generator(batch_generator, samples_per_epoch=25600, nb_epoch=1, verbose=1)

    model.save_weights('model.h5')
    json = model.to_json()
    with open('model.json', 'w') as out:
        out.write(json)