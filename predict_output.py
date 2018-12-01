import conaf
import data_generator
from keras.models import load_model
import pickle
import numpy as np
from skimage.exposure import adjust_sigmoid
from skimage import feature
from skimage.color import rgb2gray
def predict_outputs(model, val_dataset):
    val_labels = val_dataset.labels_list()
     
    all_indices = range(len(val_labels))
    
    print("_A_A_A_A_S_AS_D_S_D_AF_ADS_FA_D_SF_A_F_ASF_ASF__A_")
    print(all_indices)
    all_masks = []
    all_predicted_labels = []

    for i in all_indices:
        img = val_dataset.load_image(i)
        threshold_image = adjust_sigmoid(img)
        edge_detected = feature.canny(rgb2gray(threshold_image), sigma = 0.5)
        repeated_edge = np.repeat(edge_detected.reshape(1,1024, 1024), 3, axis = 2)

        print(i)
        output = model.predict(repeated_edge.reshape((1,1024,1024,3)))
        #all_masks.append(output[1])
        all_predicted_labels.append(output[0])

    return val_labels, all_predicted_labels

def main():
    
    model = load_model("completemodel_weights-improvement-14-0.42.hdf5")
    with open("data/dataset_val.obj", "rb") as f:
        val = pickle.load(f)
    
    l, pred = predict_outputs(model, val)
    np.save("labels.np", l)
    np.save("predictions.np", pred)

if __name__ == '__main__':
    main()


