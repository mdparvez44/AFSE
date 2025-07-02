import os
import cv2
import face_recognition
import pickle

path = 'images'
images = []
classNames = []

myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodeList.append(enc[0])
    return encodeList

encodeListKnown = findEncodings(images)

with open('encodings.p', 'wb') as f:
    pickle.dump((encodeListKnown, classNames), f)

print("✅ Encodings saved to 'encodings.p'")
