import os
import cv2
import requests
import numpy as np
import face_recognition
from os.path import splitext

# make a list of all the available images
images = os.listdir('images')
known_face_encodings = []
known_face_names = []

for i in os.listdir('images'):
    pic = face_recognition.load_image_file('images/'+i)
    known_face_encodings.append(face_recognition.face_encodings(pic)[0])
    known_face_names.append(i[:i.rfind('_')])

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

video_capture = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:

    # load your image
    _, frame = video_capture.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    qray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(qray, 1.3, 5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image_to_be_matched_encoded = face_recognition.face_encodings(frameRGB)
    cv2.imwrite('raw_img/1.jpg', frameRGB)
    if len(image_to_be_matched_encoded)>0:
        for (i, image) in enumerate(known_face_encodings):
            # encode the loaded image into a feature vector
            current_image_encoded = image_to_be_matched_encoded[0]
            # match your image with the image and check if it matches
            result = face_recognition.compare_faces([image], current_image_encoded)
                # check if it was a match
            if distance <= 0.40:

                # Takes same username without .jpg/.png Extension
                user_name = splitext(known_face_names[i])[0]

                # Shows the username to the terminal
                print("\n----OpenCV List Data----\n- Name: {}".format(user_name))

                # Django API receives the Matched username from OpenCV
                URL = "http://127.0.0.1:8000/matched-user/{}/".format(user_name)

                # Sending API get request and saving the response as response object
                data_send = requests.get(url=URL)

                # Shows the RESPONSE ro the terminal based on Status
                if data_send.status_code == 200:
                    print('- Already Exist! ', data_send)
                elif data_send.status_code == 201:
                    print('- Created :) ', data_send)
                else:
                    print('- Not Found :o', data_send)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the WebCam
video_capture.release()
cv2.destroyAllWindows()
