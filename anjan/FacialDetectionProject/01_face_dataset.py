''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

import cv2
import sys
import os
import uuid 
import names
import random
import numpy as np 


# The gender model architecture
# https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
GENDER_MODEL = 'weights/deploy_gender.prototxt'
# The gender model pre-trained weights
# https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
GENDER_PROTO = 'weights/gender_net.caffemodel'
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the gender classes
GENDER_LIST = ['Male', 'Female']
# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "weights/deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)


def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

def predict_gender(input_path: str):
    """Predict the gender of the faces showing in the image"""
    # Read Input Image
    img = cv2.imread(input_path)
    # resize the image, uncomment if you want to resize the image
    # img = cv2.resize(img, (frame_width, frame_height))
    # Take a copy of the initial image and resize it
    frame = img.copy()
    # if frame.shape[1] > frame_width:
    #     frame = image_resize(frame, width=frame_width)
    # predict the faces
    faces = get_faces(frame)
    # Loop over the faces detected
    # for idx, face in enumerate(faces):
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y: end_y, start_x: end_x]
        # image --> Input image to preprocess before passing it through our dnn for classification.
        # scale factor = After performing mean substraction we can optionally scale the image by some factor. (if 1 -> no scaling)
        # size = The spatial size that the CNN expects. Options are = (224*224, 227*227 or 299*299)
        # mean = mean substraction values to be substracted from every channel of the image.
        # swapRB=OpenCV assumes images in BGR whereas the mean is supplied in RGB. To resolve this we set swapRB to True.
        blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]

        # Draw the box
        # label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
        # print(label)
        # yPos = start_y - 15
        # while yPos < 15:
        #     yPos += 15

        # get the font scale for this image size
        # optimal_font_scale = get_optimal_font_scale(label,((end_x-start_x)+25))
        # box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        # cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        # # Label processed image
        # cv2.putText(frame, label, (start_x, yPos),
        #             cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, box_color, 2)

        # Display processed image
    #display_img("Gender Estimator", frame)
    # uncomment if you want to save the image
    # cv2.imwrite("output.jpg", frame)
    # Cleanup
    #cv2.destroyAllWindows()
    return gender, gender_confidence_score


no_gender = True

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier(sys.path[0] + '/haarcascade_frontalface_default.xml')


print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()

    
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if no_gender:
        tmp_im_path = sys.path[0] + "/dataset/tmp.jpg"
        cv2.imwrite(tmp_im_path, img)    

        gender, gconf = predict_gender(tmp_im_path)
        print(gender, gconf)    

        # write config file for name and id
        user_id = str(random.randint(1000,9999)) 
        user_nm = names.get_first_name(gender=gender)

        print(user_nm)

        with open(sys.path[0] + "/config.txt", "a") as wr:
            output = user_id + "." + user_nm + "\n"
            wr.write(output)

        face_id = user_id # names.get_first_name(gender=gender) #+ "-" + str(random.randint(1000,9999))  #uuid.uuid4() #input('\n enter user id end press <return> ==>  ')
        os.remove(tmp_im_path)
        no_gender = False


    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)


        (w_lab, h_lab), _ = cv2.getTextSize("training ...", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        img = cv2.rectangle(img, (x, y - 20), (x + w, y), [255,0,0], -1)
        img = cv2.putText(img, "training ...", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,255], 1)
        
        count += 1

        # Save the captured image into the datasets folder
        im_path = sys.path[0] + "/dataset/User." + str(face_id) + '.' + str(count) + ".jpg"
        cv2.imwrite(im_path, gray[y:y+h,x:x+w])
        cv2.imshow('image', img)



    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 30 face sample and stop video
         break





import cv2
import numpy as np
from PIL import Image
import os
import sys 

# Path for face image database
path = sys.path[0] + '/dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(sys.path[0] + "/haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        # print(imagePath)
        # input()
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids



print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write(sys.path[0] + '/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))




# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

