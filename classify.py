# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.cv2.imread(args["image"])
output = image.copy()
 
# pre-process the image for classification
image = cv2.cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# classify the input image
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# we'll mark our prediction as "correct" of the input image filename
# contains the predicted label text (obviously this makes the
# assumption that you have named your testing image files this way)
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.cv2.putText(output, label, (10, 25),  cv2.cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(label))
cv2.cv2.imshow("Output", output)
cv2.cv2.waitKey(0)





from imutils import paths
import random





























print("[INFO] testing the neural network...")


print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset/Test_Images")))

random.seed(42)
random.shuffle(imagePaths)
data=[]
labels=[]
act=[]
predict=[]
counter=0
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.cv2.imread(imagePath)
	image = cv2.cv2.resize(image, (64, 64))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
	l1 =0
	if label=="Bacteria":
		l1=1	
	elif label=="CurlVirus":
		l1=2 
	elif label=="EarlyBlight":
		l1=3
	elif label=="Healthy":
		l1=4
	elif label=="LateBlight":
		l1=5 
	elif label=="LeafMold":
		l1=6
	elif label=="Septoria":
		l1=7
	elif label=="SpiderMites":
		l1=8 
	elif label=="TargetSpot":
		l1=9
	elif label=="TomatoMosaicVirus":
		l1=10
	act.append(l1)

	image = cv2.cv2.imread(imagePath)
	output = image.copy()
 
# pre-process the image for classification
	image = cv2.cv2.resize(image, (64, 64))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	print("[INFO] classifying image...")
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]
	print("Hi")
	l2=0
	if label=="Bacteria":
		l2=1	
	elif label=="CurlVirus":
		l2=2 
	elif label=="EarlyBlight":
		l2=3
	elif label=="Healthy":
		l2=4
	elif label=="LateBlight":
		l2=5 
	elif label=="LeafMold":
		l2=6
	elif label=="Septoria":
		l2=7
	elif label=="SpiderMites":
		l2=8 
	elif label=="TargetSpot":
		l2=9
	elif label=="TomatoMosaicVirus":
		l2=10
	predict.append(l2)
	print(counter+1)
	counter=counter+1

from sklearn.metrics import confusion_matrix

op=confusion_matrix(act,predict)

print(op)

diagonalsum=0

for i in range(0,9):
	diagonalsum+=op[i][i]

totalsum=0

for i in range(0,10):
	for j in range(0,10):
		totalsum+=op[i][j]

print("ACCURACY  :::: ")
print(diagonalsum/totalsum)

precision=[0,0,0,0,0,0,0,0,0,0]
recall=[0,0,0,0,0,0,0,0,0,0]
f1score=[0,0,0,0,0,0,0,0,0,0]


for i in range(0,10):
	for j in range(0,10):
		recall[j]=recall[j]+op[i][j]
		precision[i]=precision[i]+op[i][j]


for i in range(0,10):
	recall[i]=op[i][i]/recall[i]
	precision[i]=op[i][i]/precision[i]
	f1score[i]=(2*precision[i]*recall[i])/(precision[i]+recall[i])

print ("Precision is :: ")
print (precision)

print ("Recall is ::")
print (recall)

print ("F1-Score is ::")
print (f1score)

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
print(f1_score(act, predict, average='micro'))
print(f1_score(act, predict, average='macro'))
print(f1_score(act, predict, average='weighted'))  
print(accuracy_score(act,predict))