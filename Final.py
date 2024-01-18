import cv2 as cv
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from mtcnn.mtcnn import MTCNN

import pyttsx3

from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

#development
def create_model():
    detector = MTCNN()

    class FACELOADING:
        def __init__(self, directory):
            self.directory = directory
            self.target_size = (160, 160)
            self.X = []
            self.Y = []
            self.detector = MTCNN()

        def extract_face(self, filename):
            img = cv.imread(filename)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x, y, w, h = self.detector.detect_faces(img)[0]['box']
            x, y = abs(x), abs(y)
            face = img[y:y + h, x:x + w]
            face_arr = cv.resize(face, self.target_size)
            return face_arr

        def load_faces(self, dir):
            FACES = []
            for im_name in os.listdir(dir):
                try:
                    path = dir + im_name
                    single_face = self.extract_face(path)
                    FACES.append(single_face)
                except Exception as e:
                    pass
            return FACES

        def load_classes(self):
            for sub_dir in os.listdir(self.directory):
                path = self.directory + '/' + sub_dir + '/'
                FACES = self.load_faces(path)
                labels = [sub_dir for _ in range(len(FACES))]
                print(f"Loaded successfully: {len(labels)}")
                self.X.extend(FACES)
                self.Y.extend(labels)

            return np.asarray(self.X), np.asarray(self.Y)

    faceloading = FACELOADING("C:\\Users\\ostin v thomas\\Desktop\\PROJECTS\\Face-Recognition\\Celebrities_Data")
    X, Y = faceloading.load_classes()

    embedder = FaceNet()

    def get_embedding(face_img):
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        yhat = embedder.embeddings(face_img)
        return yhat[0]

    EMBEDDED_X = []

    for img in X:
        EMBEDDED_X.append(get_embedding(img))

    EMBEDDED_X = np.asarray(EMBEDDED_X)

    # Saving Face Embeddings

    np.savez_compressed('faces_embeddings_celebritiestest.npz', EMBEDDED_X, Y)

    # Label Encoding

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    # Train-Test Data Split
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

    # SVM
    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)

    # save the model
    import pickle
    with open('svm_model_celebritiestest_160x160.pkl', 'wb') as f:
        pickle.dump(model, f)



# create_model()

#Deployment
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
previous_face_name = None  # Initialize to None at the beginning
unknown_face_welcome_given = False  # Flag to track if welcome message has been given for unknown face


def identity_check(embedding, threshold=0.5):
    global previous_face_name, unknown_face_welcome_given  # Use global variables

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

        if decoded_label[0] != "unknown" and decoded_label[0] != previous_face_name:
            welcome_message = f"Welcome to my world, {decoded_label[0]}"
            engine = pyttsx3.init()
            engine.say(welcome_message)
            engine.runAndWait()
            previous_face_name = decoded_label[0]  # Update the previous face name
            unknown_face_welcome_given = False  # Reset the flag for unknown face

        return decoded_label[0]
    else:
        if not unknown_face_welcome_given:
            welcome_message = "Welcome to my world"
            engine = pyttsx3.init()
            engine.say(welcome_message)
            engine.runAndWait()
            unknown_face_welcome_given = True  # Set the flag to indicate welcome message has been given

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


        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
        cv.putText(frame, str(face_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows()
