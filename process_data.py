import csv
import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

col_size, row_size = 64, 64

data_path = "data/"
csv_path = data_path + "driving_log.csv"

def get_data(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)    # Skip headers
        return [r for r in reader]

def process_image(image):
    # Convert to float32 for brightness augmentation
    image_32 = np.float32(image)
    # Removing the top 45 pixels of the images to remove the sky.
    # Also removing the bottom 25 pixels to remove the dash of the car.
    cropped_image = image_32[45:135, :, :]
    resized_image = cv2.resize(cropped_image, (col_size, row_size), interpolation=cv2.INTER_AREA)
    return resized_image

def get_image(path):
    raw_image = cv2.imread(path)
    return process_image(raw_image)

def save_data(pickle_filename, data):
    filepath = data_path + pickle_filename
    if not os.path.isfile(filepath):
        print('Saving data to pickle file...')
        try:
            print(filepath)
            with open(filepath, 'wb') as pfile:
                pickle.dump(data, pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_filename, ':', e)
            raise
    print('Data cached in pickle file.')



if __name__ == "__main__":

    data = get_data(csv_path)

    data_len = len(data)
    num_features = data_len * 3   # left, right, center image per data row

    images = np.empty((num_features, row_size, col_size, 3), dtype=np.float32)
    labels = np.empty(num_features)

    for i in tqdm(range(data_len)):
        row = data[i]
        steer_angle = float(row[3])

        # Get Images
        center_image = get_image(data_path + row[0].strip())
        left_image = get_image(data_path + row[1].strip())
        right_image = get_image(data_path + row[2].strip())

        # Get feature indices
        center_index = i*3
        left_index = i*3 + 1
        right_index = i*3 + 2

        # Set features
        images[center_index] = center_image
        images[left_index] = left_image
        images[right_index] = right_image

        # Set labels
        labels[center_index] = steer_angle
        labels[left_index] = steer_angle + 0.25     # Add 0.25 to steer_angle to drive back toward center
        labels[right_index] = steer_angle - 0.25    # Subtract 0.25 from steer angle to drive back toward center


    # Print out data summary
    train_size = images.shape[0]
    image_shape = images.shape[1:]

    print("Train size:", train_size)
    print("Image shape:", image_shape)

    image_filename1 = "image_data1.pickle"
    image_filename2 = "image_data2.pickle"
    label_filename = "label_data.pickle"

    # Bug in OS X w/ pickle makes saving data over 2GB fail.  Break into multiple chunks.
    for i, image_n in enumerate(np.split(images, 2)):
        save_data(data_path + "image_data" + str(i) + ".pickle", image_n)
    save_data(data_path + label_filename, labels)