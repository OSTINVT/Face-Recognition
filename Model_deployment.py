import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN


# Initialize FaceNet
facenet = FaceNet()

# Load face embeddings and labels
faces_embeddings = np.load("faces_embeddings_celebrities.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

mtcnn_model = MTCNN()

# Load SVM model
model = pickle.load(open("svm_model_celebrities_160x160.pkl", 'rb'))

# Inside the identity_check function
def identity_check(embedding, threshold=0.5):
    embedding = np.array(embedding)
    probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
    max_prob_index = np.argmax(probabilities)

    if probabilities[max_prob_index] >= threshold:
        predicted_label = model.classes_[max_prob_index]
        print("Predicted Label:", predicted_label)
        # Convert the scalar label to a 1D array
        predicted_label_array = np.array([predicted_label])
        decoded_label = encoder.inverse_transform(predicted_label_array)
        print("Decoded Label:", decoded_label)
        return decoded_label[0]
    else:
        return "unknown"




# Capture from webcam
cap = cv.VideoCapture(0)


# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = mtcnn_model.detect_faces(rgb_img)
    for result in results:
        x, y, w, h = result['box']
        img = rgb_img[y:y + h, x:x + w]
        img = cv.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = identity_check(ypred)
        print("Predicted Name:", face_name)  # Add this line
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        cv.putText(frame, str(face_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows()
