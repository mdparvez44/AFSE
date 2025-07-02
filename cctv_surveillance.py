import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
import pickle

# Path to known faces
path = 'images'
images = []
classNames = []
myList = os.listdir(path)

# Load and encode known images
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to find encodings of known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:  # Only if face is detected
            encodeList.append(enc[0])
    return encodeList

# Function to mark attendance and log time
def markAttendance(name):
    timeNow = datetime.now()
    timeString = timeNow.strftime('%H:%M:%S')

    # Create new row as DataFrame
    new_row = pd.DataFrame([[name, timeString]], columns=['Name', 'Time'])

    # Read existing file or create if not exists
    try:
        df = pd.read_csv('Attendance.csv')
        df = pd.concat([df, new_row], ignore_index=True)
    except FileNotFoundError:
        df = new_row

    df.to_csv('Attendance.csv', index=False)

# Load encodings for known faces
encodeListKnown = findEncodings(images)
print('Encoding complete.')

# Load log or create new log if not exists
log_path = 'logs/surveillance_log.csv'
if os.path.exists(log_path):
    log_df = pd.read_csv(log_path)
else:
    log_df = pd.DataFrame(columns=['Name', 'First_Seen', 'Last_Seen'])

unknown_count = len(os.listdir('unknown_faces'))

# Use CCTV feed or webcam
cap = cv2.VideoCapture(0)  # Replace with CCTV file like "cctv_footage.mp4" if needed

while True:
    success, img = cap.read()
    if not success:
        break

    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgRGB = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceLocs = face_recognition.face_locations(imgRGB)
    encodes = face_recognition.face_encodings(imgRGB, faceLocs)

    for encodeFace, faceLoc in zip(encodes, faceLocs):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        name = "Unknown"
        if len(faceDis) > 0:
            best_match = np.argmin(faceDis)
            if matches[best_match]:
                name = classNames[best_match]

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # For known faces
        if name != "Unknown":
            if name in log_df['Name'].values:
                log_df.loc[log_df['Name'] == name, 'Last_Seen'] = current_time
            else:
                log_df = pd.concat([log_df, pd.DataFrame([[name, current_time, current_time]], columns=['Name', 'First_Seen', 'Last_Seen'])], ignore_index=True)
            markAttendance(name)  # Mark attendance

        # For unknown faces
        else:
            # Process unknown faces
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            face_img = img[y1:y2, x1:x2]

            # Generate a unique filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unknown_name = f"Unknown_{unknown_count + 1}_{timestamp}.jpg"
            
            # Save the face image in 'unknown_faces' folder
            cv2.imwrite(f"unknown_faces/{unknown_name}", face_img)

            # If the unknown person is already in the log, update the 'Last_Seen'
            if unknown_name in log_df['Name'].values:
                log_df.loc[log_df['Name'] == unknown_name, 'Last_Seen'] = current_time
            else:
                # Otherwise, add them with 'First_Seen' as current time
                log_df = pd.concat([log_df, pd.DataFrame([[unknown_name, current_time, current_time]], columns=['Name', 'First_Seen', 'Last_Seen'])], ignore_index=True)

            unknown_count += 1

    cv2.imshow('CCTV', img)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the final log to CSV
log_df.to_csv(log_path, index=False)
cap.release()
cv2.destroyAllWindows()