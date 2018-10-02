# Pneumonia Detection

# Loading Data
1. `git clone https://www.github.com/matterport/Mask_RCNN.git` into the directory.
2. Change Directory to Mask_RCNN, in visualize.py, models.py and alter the line:
   ``from mrcnn import utils` to `from . import utils``
3. Download the Kaggle Dataset. Your directory should look as follows:
      ```
       .
       ├── _data_manipulation.py
       ├── _data/
       |   ├── _stage_1_train_images/
       |   ├── _stage_1_train_labels.csv
       |   ...
       ├── _Mask_RCNN
       |   ├── _models.py
       |   ├── _visualize.py
       |   ...
       ```
4. Use data_manipulation.py as necessary. Load files and their annotations with ``data_annotation.split_dataset()`` and then ``data_annotation.load_dataset()``.
