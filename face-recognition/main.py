import os
import cv2
import datetime
import numpy as np
from time import sleep
from funct_thread import FunctionThread

# function to detect face using OpenCV
import requests
from PIL import Image
from resizeimage import resizeimage

marc_url = 'http://8ba96d50.ngrok.io/video.mjpg'  # 1
juan_url = 'http://6cba3f94.ngrok.io/video.mjpg'  # 2
sergio_url = 'http://23474fb4.ngrok.io/video.mjpg'  # 3


def detect_face(img):
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # let's detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]


# function to detect face using OpenCV
def detect_faces(img):
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # let's detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces are detected then return original img
    face_parts = [(gray[y:y + w, x:x + h], (x, y, w, h)) for (x, y, w, h) in faces]

    return face_parts


def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containing images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # detect face
            face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    return faces, labels


# subjects = ["", "Marc", "Juan", "Sergio", "Edu", "Juanjo"]
subjects = ["", "Marc", "", "", "Edu", ""]


def resize_images(dir, width, height):
    s_dir = 'training-data/' + dir + '/'
    for image_file in os.listdir(s_dir):
        with open(s_dir + image_file, 'r+b') as file:
            with Image.open(file) as image:
                thumbnail = resizeimage.resize('thumbnail', image, [width, height])
                thumbnail.mode = 'RGB'
                thumbnail.save(s_dir + image_file, image.format)


def find_min_size(photos_path):
    local_min_size = (float('inf'), float('inf'))
    for dir in os.listdir(photos_path):
        s_dir = photos_path + dir + '/'
        for image_file in os.listdir(s_dir):
            with open(s_dir + image_file, 'r+b') as file:
                with Image.open(file) as image:
                    if local_min_size[0] > image.size[0] and local_min_size[1] > image.size[1]:
                        local_min_size = image.size
    return local_min_size


print("Preparing data...")

# min_size = find_min_size('training-data/')
# for f in os.listdir('training-data/'):
#     resize_images(f, min_size[0], min_size[1])
faces, labels = prepare_training_data("training-data")

print("Data prepared")
# print total faces and labels
# print("Total faces: ", len(faces))

# print("Total labels: ", len(labels))

face_recognizer = cv2.createLBPHFaceRecognizer()
face_recognizer.train(faces, np.array(labels))


# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img):
    # make a copy of the image as we don't want to change original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)
    # predict the image using our face recognizer
    label, accuracy = face_recognizer.predict(face)
    # print label, accuracy
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img


print("Predicting images...")

# load test images
# test_img1 = cv2.imread("test-data/s1/2.jpg")
# test_img2 = cv2.imread("test-data/s2/1.jpg")
# test_img3 = cv2.imread("test-data/s3/1.jpg")

# perform a prediction
# predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)
# predicted_img3 = predict(test_img3)
# print("Prediction complete")

# display both images
# cv2.imshow(subjects[1], predicted_img1)
# cv2.imshow(subjects[2], predicted_img2)
# cv2.imshow(subjects[3], predicted_img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap1 = cv2.VideoCapture(sergio_url)
cap2 = cv2.VideoCapture(marc_url)
# cap = cv2.VideoCapture(sergio_url)
cap1.set(5, 10)
cap1.set(3, 640)  # WIDTH
cap1.set(4, 480)  # HEIGHT
cap2.set(5, 10)
cap2.set(3, 640)  # WIDTH
cap2.set(4, 480)  # HEIGHT

# captures = [cv2.VideoCapture(sergio_url)]
# for capture in captures:
#     capture.set(3, 640)  # WIDTH
#     capture.set(4, 480)  # HEIGHT

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


# eye_cascade = cv2.CascadeClassifier('/haarcascades/haarcascade_eye.xml')

def stateChanges(previous_labels_in_picture, labels_in_picture, light, light_old):
    if len(previous_labels_in_picture) != len(labels_in_picture):
        return True
    else:
        for idx, label in enumerate(previous_labels_in_picture):
            if label != labels_in_picture[idx]:
                return True
    if light != light_old:
        return True
    return False


previous_labels_in_picture1 = []
previous_labels_in_picture2 = []


# previous_labels_in_pictures = []
# for cap in captures:
#     previous_labels_in_pictures.append([])

def send_request(labels_in_pic, light, room):
    lvalue = 'OFF'
    if light:
        lvalue = 'ON'
    labels = ""
    for idx, label in enumerate(labels_in_pic):
        if idx != 0:
            labels += ", "
        labels += label
    requests.post('https://efi-home-sergiowalls.c9users.io:8080/rooms/' + str(room) + '/events',
                  data={"count": len(labels_in_pic), "names": labels, "light": lvalue})


count_frames1 = 0
count_frames2 = 0
light1 = False
light2 = False
light1_old = False
light2_old = False

while True:
    ret1, frame1 = cap1.read()
    img1 = frame1.copy()
    if count_frames1 == 10:
        count_frames1 = 0
        faces1 = detect_faces(frame1)

        # print len(faces1)
        labels_in_picture1 = []

        for (face_img1, (x, y, w, h)) in faces1:
            label1, accuracy1 = face_recognizer.predict(face_img1)
            label_text1 = subjects[label1]
            labels_in_picture1.append(label_text1)

            draw_rectangle(img1, (x, y, w, h))
            draw_text(img1, label_text1, x, y - 5)

        if stateChanges(previous_labels_in_picture1, labels_in_picture1, light1, light1_old):
            # previous_labels_in_pictures[idx] = labels_in_picture
            light1_old = light1
            previous_labels_in_picture1 = labels_in_picture1
            FunctionThread(send_request, labels_in_picture1, light1, 1).start()
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cv2.imshow("Frame 1", img1)
    else:
        count_frames1 += 1
    # cv2.imshow("Frame " + str(idx), img)

    ret2, frame2 = cap2.read()
    img2 = frame2.copy()
    if count_frames2 == 10:
        count_frames2 = 0
        faces2 = detect_faces(frame2)

        # print len(faces2)
        labels_in_picture2 = []

        for (face_img2, (x, y, w, h)) in faces2:
            label2, accuracy2 = face_recognizer.predict(face_img2)
            label_text2 = subjects[label2]
            labels_in_picture2.append(label_text2)

            draw_rectangle(img2, (x, y, w, h))
            draw_text(img2, label_text2, x, y - 5)

        if stateChanges(previous_labels_in_picture2, labels_in_picture2, light2, light2_old):
            # previous_labels_in_pictures[idx] = labels_in_picture
            light2_old = light2
            previous_labels_in_picture2 = labels_in_picture2
            FunctionThread(send_request, labels_in_picture2, light2, 2).start()
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cv2.imshow("Frame 2", img2)
    else:
        count_frames2 += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('n'):
        light1 = True
        print "light1 ON"
        # requests.post('https://efi-home-sergiowalls.c9users.io:8080/rooms/1/events', data={"count": 0, "light": 'ON'})
    elif cv2.waitKey(1) & 0xFF == ord('m'):
        light1 = False
        print "light1 OFF"
    elif cv2.waitKey(1) & 0xFF == ord('j'):
        light2 = True
        print "light2 ON"
        # requests.post('https://efi-home-sergiowalls.c9users.io:8080/rooms/2/events', data={"count": 0, "light": 'ON'})
    elif cv2.waitKey(1) & 0xFF == ord('k'):
        light2 = False
        print "light1 OFF"

# When everything done, release the capture
# for cap in captures:
#     cap.release()
cap1.release()
cap2.release()

cv2.destroyAllWindows()
