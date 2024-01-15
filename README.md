# Face-Recognition

# Overview:
This repository contains a collection of celebrity images and the associated code for developing and deploying a face recognition model. The celebrities included in the dataset are Cristiano Ronaldo, Emma Watson, Kylian Mbappe, Lionel Messi, Neymar Jr, Tom Holland, Virat Kohli, and Zendaya.

The face recognition pipeline involves using MTCNN for face detection, FaceNet for generating face embeddings, and Support Vector Machine (SVM) for classification.

# Files:

# Celebrities_Data
This directory includes images of the following celebrities:

Cristiano Ronaldo
Emma Watson
Kylian Mbappe
Lionel Messi
Neymar Jr
Tom Holland
Virat Kohli
Zendaya
# Final.py
This file, Final.py, is the combined code for both model development and deployment. It serves as the main script that integrates the functionalities of both stages.

# Model_deployment.py
The Model_deployment.py file contains the code responsible for deploying the face recognition model. It includes the necessary code to make the model accessible and usable in a production environment.

# Model_development.py
In Model_development.py, you can find the code used for developing the face recognition model. This file encompasses the training and evaluation procedures.

# faces_embeddings_celebrities.npz
This file, faces_embeddings_celebrities.npz, stores the face embeddings of celebrities. These embeddings are extracted using FaceNet, a facial recognition model.

# svm_model_celebrities_160x160.pkl
The file svm_model_celebrities_160x160.pkl contains the trained Support Vector Machine (SVM) model. This model is utilized for classifying and recognizing faces based on the embeddings generated by FaceNet.

# How to Use:
Data Collection: Ensure that the Celebrities_Data directory contains images of the specified celebrities.

Model Development: If you wish to retrain the model or modify its architecture, refer to the Model_development.py file. Adjust the code as needed and run the script to generate updated model files.

Model Deployment: For deploying the model in a production environment, use the Model_deployment.py file. This script contains the necessary code to set up the model for real-time face recognition.

Running the Combined Code: If you prefer a one-step process, use the Final.py script. This file combines both model development and deployment functionalities.

Pre-trained Models: If you only want to use the pre-trained model without modifications, make sure to keep the faces_embeddings_celebrities.npz and svm_model_celebrities_160x160.pkl files intact.

# Acknowledgments:
The face recognition model in this repository utilizes MTCNN for face detection, FaceNet for generating face embeddings, and SVM for classification. The labeled celebrity face embeddings are used to train the SVM classifier.

Feel free to explore and adapt the code according to your needs!
