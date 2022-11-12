import cv2
import sys
import os
import uuid 
import names
import random
import pathlib
import pickle

from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot


# ***********************

### CHECK FOR INFERENCE

# ***********************






# ***********************

### TRAINING

# ***********************




# facedetect.py
# ***********************

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face

    try:
        x1, y1, width, height = results[0]['box']
    except:
        print(results)
        input()
        return None
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)

        if face is not None:
            # store
            print(path)
            faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            print("No directory exists")
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# ***********************



# faceembed.py
# ***********************

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# ***********************



cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier(sys.path[0] + '/haarcascade_frontalface_default.xml')


# For each person, enter one numeric face id
gender  = "male"

# write config file for name and id
user_id = str(random.randint(1000,9999)) 
user_nm = names.get_first_name(gender=gender)


# **** #


# with open(sys.path[0] + "/config.txt", "a") as wr:
#     output = user_id + "." + user_nm + "\n"
#     wr.write(output)

face_id = user_id # names.get_first_name(gender=gender) #+ "-" + str(random.randint(1000,9999))  #uuid.uuid4() #input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        (w_lab, h_lab), _ = cv2.getTextSize("training ...", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        img = cv2.rectangle(img, (x, y - 20), (x + w, y), [255,0,0], -1)
        img = cv2.putText(img, "training ...", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,255], 1)
        
        count += 1

        # Save the captured image into the datasets folder
        img_output_file = sys.path[0] + "/dataset/train/"+ str(user_nm) +"/User." + str(face_id) + '.' + str(count) + ".jpg"
        pp = pathlib.Path(img_output_file)
        pp.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(img_output_file, gray[y:y+h,x:x+w])
        cv2.imshow('image', img)




    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 20 and count < 50: # Take 30 face sample and stop video
        break
        
        # TRAIN IF NOT TRAINED

        # cam.release()
        # cv2.destroyAllWindows()        

        # extract_face & save dataset_train.npz
        trainX, trainy = load_dataset(sys.path[0] + "/dataset/train/") 
        print(trainX.shape, trainy.shape)     
        savez_compressed(sys.path[0] + '/dataset_train.npz', trainX, trainy)  


        # get_embedding & save embedding_train.npz 
        # load the face dataset
        data = load(sys.path[0] + '/dataset_train.npz')
        trainX, trainy = data['arr_0'], data['arr_1']

        model = load_model(sys.path[0] + '/facenet_keras.h5')
        print('Loaded Model')


        print("***********************************************************")
        # convert each face in the train set to an embedding
        newTrainX = list()
        for face_pixels in trainX:
            embedding = get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        print(newTrainX.shape)
        savez_compressed(sys.path[0] + '/embeddings_train.npz', newTrainX, trainy)




        # train the model 
        # load face embeddings
        data = load(sys.path[0] + '/embeddings_train.npz')
        trainX, trainy = data['arr_0'], data['arr_1']

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)

        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)

        print("***********************************************************")

        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy) 



        #model_save_path = ""
        #model.save(sys.path[0] + '/facenet_keras.h5')      


        with open(sys.path[0] + '/model.pkl','wb') as f:
            pickle.dump(model, f)         
        break


cam.release()
#cv2.destroyAllWindows()

# INFER
count = 0
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

while True:

    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
       )

    for(x,y,w,h) in faces:

        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        count += 1

        # Save the captured image into the datasets folder
        img_output_file = sys.path[0] + "/dataset/val/"+ str(user_nm) +"/User." + str(face_id) + '.' + str(count) + ".jpg"
        pp = pathlib.Path(img_output_file)
        pp.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(img_output_file, gray[y:y+h,x:x+w])
        cv2.imshow('image', img)       

    if count > 2:
        # extract_face & save dataset_train.npz
        testX, testy = load_dataset(sys.path[0] + "/dataset/val/")    
        print(trainX.shape, trainy.shape) 
        savez_compressed('dataset_test.npz', testX, testy)  


        # get_embedding & save embedding_train.npz 
        # load the face dataset
        data = load('dataset_test.npz')
        testX, testy = data['arr_0'], data['arr_1'] 

        dataxx = load('dataset.npz')
        testX_faces = dataxx['arr_0']

        model = load_model(sys.path[0] + '/facenet_keras.h5')
        #print('Loaded Model')

        #model = SVC(kernel='linear', probability=True) # load local model
        #model.fit(trainX, trainy)


        print("***********************************************************")
        # convert each face in the train set to an embedding
        newTrainX = list()
        for face_pixels in testX:
            embedding = get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        print(newTrainX.shape)
        savez_compressed(sys.path[0] + '/embeddings_train.npz', newTrainX, trainy)




        random_face_pixels = testX_faces[0]
        random_face_emb = testX[0]
        random_face_class = testy[0]
        random_face_name = out_encoder.inverse_transform([random_face_class])
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        print('Expected: %s' % random_face_name[0])

while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        (w_lab, h_lab), _ = cv2.getTextSize("...", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        img = cv2.rectangle(img, (x, y - 20), (x + w, y), [255,0,0], -1)
        img = cv2.putText(img, "...", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,255], 1)
        
        count += 1

        # Save the captured image into the datasets folder
        img_output_file = sys.path[0] + "/dataset/val/"+ str(user_nm) +"/User." + str(face_id) + '.' + str(count) + ".jpg"
        pp = pathlib.Path(img_output_file)
        pp.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(img_output_file, gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

        


    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count > 2:
        # extract_face & save dataset_train.npz
        testX, testy = load_dataset(sys.path[0] + "/dataset/val/")    
        print(trainX.shape, trainy.shape) 
        savez_compressed('dataset_test.npz', testX, testy)  


        # get_embedding & save embedding_train.npz 
        # load the face dataset
        data = load('dataset_test.npz')
        testX, testy = data['arr_0'], data['arr_1']

        model = load_model('facenet_keras.h5')
        print('Loaded Model')

        # convert each face in the test set to an embedding
        newTestX = list()
        for face_pixels in testX:
            embedding = get_embedding(model, face_pixels)
            newTestX.append(embedding)
        newTestX = asarray(newTestX)
        print(newTestX.shape)
        savez_compressed('embeddings_test.npz', newTestX, testy)

        # train the model 
        # load face embeddings
        data = load('embeddings_test.npz')
        testX, testy = data['arr_0'], data['arr_1']

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        testX = in_encoder.transform(testX)

        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(testy)
        testy = out_encoder.transform(testy)



        # test model on a random example from the test dataset
        selection = choice([i for i in range(testX.shape[0])])
        random_face_pixels = testX_faces[selection]
        random_face_emb = testX[selection]
        random_face_class = testy[selection]
        random_face_name = out_encoder.inverse_transform([random_face_class])
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        print('Expected: %s' % random_face_name[0])


# Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
# cam.release()
# cv2.destroyAllWindows()
