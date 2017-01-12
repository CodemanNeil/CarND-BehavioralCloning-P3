from keras.models import model_from_json
from keras.optimizers import Adam
from tqdm import tqdm

from model import training_batch_generator
from process_data import get_data, get_image
import numpy as np

live_csv_path = "train_data/driving_log.csv"
row_size, col_size = 64, 64

if __name__ == "__main__":
    data = get_data(live_csv_path)
    num_features = len(data)
    images = np.empty((num_features, row_size, col_size, 3), dtype=np.float32)
    labels = np.empty(num_features)
    print(num_features)
    for i in tqdm(range(num_features)):
        row = data[i]
        steer_angle = float(row[3])
        image = get_image(row[0].strip())
        images[i] = image
        labels[i] = steer_angle

    model_filename = "model.json"
    with open(model_filename, 'r') as jfile:
        # instead.
        model = model_from_json(jfile.read())

    model.compile(Adam(lr=0.00001), "mse")
    weights_file = model_filename.replace('json', 'h5')
    model.load_weights(weights_file)

    batch_size = 256
    batch_generator = training_batch_generator(images, labels, batch_size)

    model.fit_generator(batch_generator, samples_per_epoch=2560, nb_epoch=1, verbose=1)

    model.save_weights('model_update.h5')
    json = model.to_json()
    with open('model_update.json', 'w') as out:
        out.write(json)