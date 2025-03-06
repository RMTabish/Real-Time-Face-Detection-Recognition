import cv2
import numpy as np
import face_recognition
import os 

path="faces"

#store the image object and the image name 
images=[]
classNames=[]

#iterate over the images in the folder and add to the list 
for  img in os.listdir(path):
    image=cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])

print(classNames)

#encode the face image 
def encode_face(images):
    encodings=[]

    for img in images:
        #dlib use the rgb format 
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        encode =face_recognition.face_encodings(img)[0]
        print(encode)

        encodings.append(encode)
    return encodings

knownEncodes=encode_face(images)
print("Encoding done")