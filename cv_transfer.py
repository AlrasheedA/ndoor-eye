#Arrays 
import numpy as np

#Image processing 
from scipy.misc import imresize
from imageio import imread 

#For loading files from directories and splitting data 
from sklearn.datasets import load_files

#Keras for deep learning 
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint  
from keras import applications

#For kfold cross validation 
from sklearn.model_selection import StratifiedKFold
import pickle


#Functions for reading paths and returning tensors 
def path_to_tensor(img_path):
	img = imread(img_path)
	img = imresize(img, (256,256,3))
	return np.expand_dims(img, axis=0)

def paths_to_tensors(paths):
	"""
	Input: collection of image paths of length nb 
	Outputs: (nb, 256, 256, 3) normalized tensor 
	"""
	return (np.vstack([path_to_tensor(img_path) for img_path in paths])).astype('float32')/255



inception = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
#Freeze existing layers
for layer in inception.layers:
    layer.trainable = False



#Read resized images paths and store to input X and target y 
temp = load_files('resized-data', load_content=False, categories=['Doors','Stairs','Sign'])
X = temp['filenames']
y = np.array(temp['target'])
X_tensors = paths_to_tensors(X)
y_cat = np_utils.to_categorical(y)


#Set up folds and results variable
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(kfold.split(X, y))
transfer_cv_results = []
pickle.dump(folds, open('folds','wb'))

#Run the cross validation by looping over folds' train/test indexes 
counter = 0
for train, test in folds:
	counter+=1
	
	x = inception.output
	x = GlobalAveragePooling2D()(x)
	pred = Dense(3, activation='softmax')(x)

	transfer_model = Model(inputs=inception.input, outputs = pred)

	#Get inception with weights trained on ImageNet dataset
	transfer_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
				metrics=['accuracy', 'categorical_accuracy'])

	transfer_model.fit(X_tensors[train], y_cat[train], epochs=3, batch_size=20, verbose=0)
	
	transfer_cv_results.append(transfer_model.evaluate(X_tensors[test], y_cat[test], verbose=0))
	print ('Finished %i folds'%counter)
	
	#Save the folds and results in pickle file 
	pickle.dump(transfer_cv_results, open('transfer_cv_results','wb'))


print ('Completed Job')
