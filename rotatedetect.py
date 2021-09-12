import cv2
import numpy as np
from os import listdir, walk
from os.path import isfile, join



# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def rotateImage(x1, y1, x2, y2, image):
    angle = math.atan((y2-y1)/(x2-x1))
    center = (int((x1+x2)/2), int((y1+y2)/2))
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(src = image, M=rotate_matrix, dsize=(x2-x1, y2-y1))
    return rotated_image

def detectFaceEyes(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    eyect = 0
    facect = 0
    scale_factor = 1.05
    min_neighbors = 4
    while (eyect != 2 or facect != 1) and min_neighbors < 50:
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        eyect = 0
        facect = len(faces)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scale_factor, min_neighbors)
            eyect += len(eyes)
            for(x2,y2,w2,h2) in eyes:
                cv2.rectangle(roi_color, (x2,y2), (x2+w2,y2+h2), (0,255,0), 2)
                centerx = int(x2+w2/2);
                centery = int(y2+h2/2);
                cv2.circle(roi_color, (centerx, centery), 1, (0, 255, 0), 2)
        # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()
    if(min_e)
    return ()
