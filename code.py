import mediapipe as mp   #cross platform library developed by google that provides amazing raedy-to-use ML solutions for CV
import numpy as np 
import cv2 
 
#cv.vediocapture(1) means second cam or web cam
#cv.vediocapture(filename.mp4) video file to read it frame by frame 
#it returns a tuple(return value,image)
#CvCapture is used instead of videocapture = cv2.VideoCapture(0) will return the video on your first web cam
#cap
cap = cv2.VideoCapture(0)  #camera

name = input("Enter the name of the data : ")

holistic = mp.solutions.holistic  # holistic solution for prediction it wil take the frame and return the landmarks a
hands = mp.solutions.hands #for showing the visuals
holis = holistic.Holistic() #class object
drawing = mp.solutions.drawing_utils #show the visuals
#to save it as a numpy array 
X = [] #will be the connection of all the rows
data_size = 0  #100*1020

while True:
	lst = [] #wil store landmarks will have 1020 columns of landmarks

	_, frm = cap.read()

	frm = cv2.flip(frm, 1)  #to avoid mirror effect

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))  #convert frame to rgb to be prepared for the process

#to add landmarks in lst
	if res.face_landmarks:
		for i in res.face_landmarks.landmark:
			lst.append(i.x - res.face_landmarks.landmark[1].x) #21 value, 1 is reference point instead of 0 
			lst.append(i.y - res.face_landmarks.landmark[1].y)#21 value

		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)  #8 is reference point instead of 0 
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
# is left hand not in the video
  	else:
			for i in range(42):
				lst.append(0.0)

		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)


		X.append(lst)
		data_size = data_size+1



	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2) #put data in frame and cinvert data size to string

	cv2.imshow("window", frm) #will show frames to the user
  #if the user press the escape key it will destroy all the windows and release the camera resourses and get out of the loop

	if cv2.waitKey(1) == 27 or data_size>99:
		cv2.destroyAllWindows()
		cap.release()
		break


np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)

#TRAINING

import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical #to convert class vector(integer) to binary values ,,here to convert words to integer

from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1

label = [] #we will convert words to integer numbers
dictionary = {} #will contain unique words key and value
c = 0

for i in os.listdir():
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  #if the file .npy
		if not(is_init):
			is_init = True 
			X = np.load(i)
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1) #0 elememnt file name only without npy,(-1,1) means data become in one column
		else:
			X = np.concatenate((X, np.load(i))) #file
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1))) #data

		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1

#conversion or encoding
for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

###  hello = 0 nope = 1 ---> [1,0] ... [0,1]

y = to_categorical(y)
#shuffle data
X_new = X.copy()
y_new = y.copy()
counter = 0  

cnt = np.arange(X.shape[0])  #this func return a row of values aranged 
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]  #the counter value will equal the shuffled value
	y_new[counter] = y[i]
	counter = counter + 1


ip = Input(shape=(X.shape[1]))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)  #connected to m and every layer connected to the previous

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50)


model.save("model.h5")
np.save("labels.npy", np.array(label))


#interface
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 


model  = load_model("model.h5")
label = np.load("labels.npy")



holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)



while True:
	lst = []

	_, frm = cap.read()

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))


	if res.face_landmarks:
		for i in res.face_landmarks.landmark:
			lst.append(i.x - res.face_landmarks.landmark[1].x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)

		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		lst = np.array(lst).reshape(-1,1)

		pred = label[np.argmax(model.predict(lst))]

		print(pred)
		cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

		
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break

