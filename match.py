import cv2
import numpy as np
import face_recognition
import os 

path = "faces"

# Store images and class names
images = []
classNames = []

# Read images from the folder
for img in os.listdir(path):
    image = cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])

print(classNames)

# Encode the face images
def encode_face(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        encodes = face_recognition.face_encodings(img)
        if encodes:  
            encodings.append(encodes[0])
    return encodings

knownEncodes = encode_face(images)
print("Encoding done")

# Video Capture
scal = 0.35
box_multiplier = 1 / scal
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break  

    # Resize frame for faster processing
    new_image = cv2.resize(frame, (0, 0), None, scal, scal)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame and encode them
    faceLocations = face_recognition.face_locations(new_image, model='hog')
    face_encodes = face_recognition.face_encodings(new_image, faceLocations)

    # Compare detected faces with known encodings
    for encode, faceLocation in zip(face_encodes, faceLocations):
        matches = face_recognition.compare_faces(knownEncodes, encode, tolerance=0.6)
        faceDis = face_recognition.face_distance(knownEncodes, encode)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = "UNKNOWN"

        # Draw face bounding box
        y1, x2, y2, x1 = faceLocation
        y1, x2, y2, x1 = int(y1 * box_multiplier), int(x2 * box_multiplier), int(y2 * box_multiplier), int(x1 * box_multiplier)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

    
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
