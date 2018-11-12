import data_manipulation
import numpy as np
import skimage.transform
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, detector, list_IDs, labels, batch_size, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=False):
        'Initialization'
        self.detector = detector
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        masks = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image = self.detector.load_image(ID)
            mask, _ = self.detector.load_mask(ID)
            comMask = np.any(mask, axis = 2).astype(int)
            if image.shape[0] != self.dim:
                X[i,] = skimage.transform.resize(image, output_shape = (*self.dim, self.n_channels))
                masks[i,] = skimage.transform.resize(image, output_shape = (*self.dim))
            else:
                X[i,] = image
                masks[i,] = comMask

            # Store class
            y[i] = self.labels[ID]
        output = {'output1':keras.utils.to_categorical(y, num_classes=self.n_classes), 'output2':comMask}
        return X, output
