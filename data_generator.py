import data_manipulation
import numpy as np
import skimage.transform
import keras

class DataGenerator(keras.utils.Sequence):
    IMAGE_DIM = 1024
    SCALE = 1.2
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
            mask = getProbDist(ID, SCALE)
            if image.shape[0] != self.dim:
                X[i,] = skimage.transform.resize(image, output_shape = (*self.dim, self.n_channels))
                masks[i,] = skimage.transform.resize(mask, output_shape = self.dim)
            else:
                X[i,] = image
                masks[i,] = mask

            # Store class
            y[i] = self.labels[ID]
        output = {'output1':keras.utils.to_categorical(y, num_classes=self.n_classes), 'output2':masks}
        return X, output
    
    def probability_distribution(image_id, width, height, scale, x, y):
        center_x = int(x+width/2)
        center_y = int(y+height/2)
        max_radius = int(np.sqrt((height/2)**2 + (width/2)**2))
    
        #assume cv2 used to load image
        x_dim = IMAGE_DIM
        y_dim = IMAGE_DIM
        new_image = np.zeros(shape=(x_dim,y_dim))
        
        for xi in range (int(-scale*width/2), int(scale*width/2)):
            for yi in range (int(-scale*height/2), int(scale*height/2)):
                r = np.sqrt((xi)**2 + (yi)**2)
                probability = 1.0-(r/(scale*max_radius))
                probability = 0 if probability < 0 else probability
                x_new_image = (center_x + xi)
                y_new_image = (center_y + yi)
                if(x_new_image > -1 and x_new_image < x_dim and y_new_image > -1 and y_new_image < y_dim ):
                    new_image[x_new_image][y_new_image] = round(probability, 2) 
        
        return new_image

    def getProbDist(image_id, scale):
        pneu = 0
        info = self.detector.image_info[image_id]
        x_dim = IMAGE_DIM
        y_dim = IMAGE_DIM
        probDist = np.zeros(shape=(x_dim,y_dim))
        annotations = info['annotations']
        if(len(annotations) > 0):
            patientId = annotations[0].patientId
            for a in annotations:
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    probDist += probability_distribution(image_id, h, w, scale, y, x)
            np.clip(probDist, 0, 1)
        return probDist
