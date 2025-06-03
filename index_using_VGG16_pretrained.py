# https://youtu.be/dCcRWdmmgA0
"""
@author: DigitalSreeni

This code will read images from a specified directory and extracts features
using a pre-trained VGG16 network on Imagenet database. The VGG16 network is
imported from the accompanying file VGG_feature_extractor.py where we import
the network with the top classifier layers. The final output for the feature vector
is 512. These features from every image in the input folder are captured into
a hdf5 file. This file will be imported to the query_using_VGG16_features.py 
file to search for similar images to a query image by comparing the feature 
vectors. 

"""

import os
import h5py
import numpy as np
from PIL import Image
from VGG_feature_extractor import VGGNet

# where your .tif (and maybe other) images live
images_path = "//storage-ume.slu.se/home$/joms0005/Desktop/SLU/materials/presentations/presentation_04_06/small_set"

# initialize your VGG-based feature extractor
model = VGGNet()

feats = []
names = []

# loop through every file in images_path
for filename in os.listdir(images_path):
    if not filename.lower().endswith(".tif"):
        continue

    img_path = os.path.join(images_path, filename)
    print("Extracting features from image:", filename)

    # Pass the .tif‐path directly to extract_feat()
    # (inside, it’ll do: image.load_img(img_path, target_size=(224,224)) )
    feat_vector = model.extract_feat(img_path)

    feats.append(feat_vector)
    names.append(filename)

feats = np.array(feats)

output = "CNNFeatures.h5"
print("Writing features to HDF5 file:", output)
with h5py.File(output, "w") as h5f:
    h5f.create_dataset("dataset_1", data=feats)
    names_bytes = np.array(names, dtype="S")
    h5f.create_dataset("dataset_2", data=names_bytes)

print("Done.")

# write to HDF5
output = "CNNFeatures.h5"
print("Writing feature extraction results to HDF5 file:", output)
with h5py.File(output, "w") as h5f:
    h5f.create_dataset("dataset_1", data=feats)
    # store filenames as bytes
    names_bytes = np.array(names, dtype="S")  
    h5f.create_dataset("dataset_2", data=names_bytes)

print("Done.")


