import matplotlib
matplotlib.use('TkAgg')

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.plot()

cap = cv2.VideoCapture(0)

model=tf.keras.models.load_model('/Users/divalicious/Desktop/my_model.h5')

while(True):
	ret, img = cap.read()
	fig = plt.figure()

	#apply same face detection procedures
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')		
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		detected_face = img[int(y):int(y+h),int(x):int(x+w)]
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

		img_pixels = tf.keras.preprocessing.image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		 
		img_pixels /= 255
		 
		predictions = model.predict(img_pixels)
		 
		Y=predictions[0]
		#find max indexed array
		max_index = np.argmax(predictions[0])		 
		emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
		emotion = emotions[max_index]

		
		#### Show emotions as histograms
		emotion_analysis(Y)
		fig.canvas.draw()
		

		#### Show value as line graph
		#line1= plt.plot(emotions,Y)
		#fig.canvas.draw()

		# convert canvas to image
		img1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
		img1  = img1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
		cv2.imshow("Plot",img1)
		 
		
		#### Show value of each emotion #######
		#cv2.putText(img,"Angry:"+str(predictions[0][0]), (int(x-400), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		#cv2.putText(img,"Disgust:"+str(predictions[0][1]), (int(x-400), int(y+20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		#cv2.putText(img,"Fear:"+str(predictions[0][2]), (int(x-400), int(y+40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		#cv2.putText(img,"Happy:"+str(predictions[0][3]), (int(x-400), int(y+60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		#cv2.putText(img,"Sad:"+str(predictions[0][4]), (int(x-400), int(y+80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		#cv2.putText(img,"Surprise:"+str(predictions[0][5]), (int(x-400), int(y+100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		#cv2.putText(img,"Neutral:"+str(predictions[0][6]), (int(x-400), int(y+120)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		cv2.putText(img,emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

	cv2.imshow('Frame',img)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
