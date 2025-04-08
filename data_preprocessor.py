import os
import cv2
import numpy as np
from config import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def read_data():
	image_list = []
	label_list = []
	for folder in os.listdir(DATA_PATH):
		filepath = os.path.join(DATA_PATH, folder)
		if os.path.isdir(filepath):
			for files in os.listdir(filepath):
				image = cv2.imread(os.path.join(filepath, files), 0)
				image = image.astype("float32") / 255.0
				image_list.append(image)
				label_list.append(folder)
	
	image_list = np.array(image_list)
	label_list = np.array(label_list).reshape(-1, 1)

	enc = OneHotEncoder(sparse_output = False)
	label_list = enc.fit_transform(label_list)
	
	X_train, X_text, y_train, y_test = train_test_split(image_list, label_list, test_size = 0.2, random_state = 42)
	X_train = np.expand_dims(X_train, axis=-1)

	return X_train, X_text, y_train, y_test