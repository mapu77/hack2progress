import os
from time import sleep
import cv2

# cap = cv2.VideoCapture('http://f00b526f.ngrok.io/video.mjpg')
import datetime

marc_url = 'http://8ba96d50.ngrok.io/video.mjpg' # 1
juan_url = 'http://69265ace.ngrok.io/video.mjpg' # 2
sergio_url = 'http://23474fb4.ngrok.io/video.mjpg' # 3
# Edu: 4
# Juanjo: 5
cap = cv2.VideoCapture(marc_url)
#cap = cv2.VideoCapture(0)
cap.set(3, 640)  # WIDTH
cap.set(4, 480)  # HEIGHT

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/haarcascades/haarcascade_eye.xml')

i = 0
label = 4

if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + '/training-data'):
    os.mkdir(os.path.dirname(os.path.abspath(__file__)) + '/training-data')
if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + '/training-data/s' + str(label)):
    os.mkdir(os.path.dirname(os.path.abspath(__file__)) + '/training-data/s' + str(label))

timeStart = datetime.datetime.now()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces), i)
    # Display the resulting frame
    dif_time = datetime.datetime.now() - timeStart
    if len(faces) == 1 and dif_time.microseconds >= 500:
        print "saved"
        timeStart = datetime.datetime.now()
        (x, y, w, h) = faces[0]
        cv2.imwrite("training-data/s" + str(label) + "/" + str(i) + ".jpg", frame[y:y + h, x:x + w])
        i += 1

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
