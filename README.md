# Portfolio

A compilation and brief description of all the AI models I have made

### Feel Free to contact me through [linkedin](https://www.linkedin.com/in/hamza-amir-0616m/) or through email hamzaamir0616@gmail.com

<br>

## Table Of Contents

The Projects are listed by order of degree of complexity

1. [Celebrity Image Recognittion](#celebrity-image-recognittion)

2. [Potato Diesease Identification](#potato-diesease-identification)


3. [Titanic Passenger survival Chance Prediction](#titanic-passenger-survival-chance-prediction)

Those listed below are of about simmilar complexity

4. [Hand Written Digits Recognition](#hand-written-digits-recognition)

5. [Bank Customer Rentention Prediction](#bank-customer-rentention-prediction)

6. [Clothing Identifier Model](#clothing-identifier-model)

<br>

## [Celebrity Image Recognittion](https://github.com/HamzaAmirr/Celebrity_Image_Recognition)

- **Objective:** developing a Computer vision system for recognizing celebrities from images using a Convolutional Neural Network (CNN).

- **Dataset:** A total of 4,267 images of 17 celebrities, collected, compiled and edited by me

- **Model Artchitechture:** The model is built using TensorFlow and Keras, using transfer learning with EfficientNetB0 paired with a dense neural network to learn distinctive features of celebrity faces. 

- **Model Evaluation and accuracy:** The performance of the model is evaluated using standard metrics including accuracy, precision, recall, and F1-score. The final model achieves a **90% accuracy** and a **loss of 0.44** on the validation set, demonstrating the model's effectiveness in correctly identifying celebrities. 

- Additionally, the project involves techniques for saving and loading the model, as well as generating predictions on new data. Finally, the results of this project are demonstrated in a web application hosted locally through FastAPI and Uvicorn. 

- This system has potential applications in automated celebrity recognition for media, entertainment, and security industries.

## [Potato Diesease Identification](https://github.com/HamzaAmirr/Potato_Disease_Identtification)

- **Objective:** A Computer Vision system for detecting and classifying potato plant disease from the picture of its leaf using a Convolutional Neural Network

- **Dataset:** The dataset consists of 2152 images of potato leaves, which are divided into training and validation sets. Another dataset of 300 images for testing. Both of these datasets were obtained from kaggle. Credits are in the project folder

- **Model Architechture:** The model is built using Tensorflow and Keras. It uses 6 standard CNN blocks along with two dense layers.

- **Model Evaluation and Accuracy:** The performance of the model is evaluated using standard metrics including accuracy, precision, recall, and F1-score. The final model achieves a **98% accuracy** and a **loss of 0.7** on the test set, demonstrating the model's effectiveness in correctly identifying celebrities.

## [Titanic Passenger survival Chance Prediction](https://github.com/HamzaAmirr/Clothing_Identifier_Model)

- **Objective:** This model was developed as the titanic-kaggle competition. It uses Machine Learning techniques to predict which passenger on the titanic would survive given certain parameters

- **Dataset:**  The data was pre splitted into train and test datasets, but just like all datasets, it required pre proprocessing. More information on the dataset can be found [here](https://www.kaggle.com/competitions/titanic/data).

- **Model Architechture:** The final model used XGBoost Classifier to achieve the task

- **Model Training:** Multiple Machine learning techniques were employed along with grid search, to reach the best parameters for each type of model. Howwever XGBoost classifier was the only model which acheived a **training accuracy of above 80% (i.e. 82%)**

- **Model Accuracy:** This model achieved an **80% accuracy on the test data** set and **around 78% accuracy** in the competition submission dataset

## [Hand Written Digits Recognition](https://github.com/HamzaAmirr/Handwritten_Digits_Recognition)
- **Objective:** This project aims to develop a Deep learning model that can recognize handwritten digits from hand-drawn
images of digits

- **Dataset:** The dataset used for this project is the MNIST dataset, which is a widely
used dataset for handwritten digit recognition. It consists of 60,000 images of
handwritten digits for training and 10,000 images for testing.

- **Model Architechture:** The final model used a single CNN block along with a dense layer to achieve the task 

- **Model Accuracy:** The model achieved an **accuracy of 99%** and a **loss of 0.04** on the test dataset.

## [Bank Customer Rentention Prediction](https://github.com/HamzaAmirr/Bank_Costumer_retention_prediction)

- **Objective:** This project aims to develop a Machine Learning model that can predict whether a bank customer
will continue to use the bank's services or not

- **Dataset:** A Dataset from kaggle was used to develop this model. The dataset contains 10,000 samples with 14 features each. The Data is contained in a csv file available [here](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

- **Model Architechture:** The Deeplearning model contains just 3 densely connected layers, inlcuding an input and output layer

- **Model Accuracy:** The model achieved an **accuracy of 86%** with a **loss of 0.34** on the test dataset.

## [Clothing Identifier Model](https://github.com/HamzaAmirr/Clothing_Identifier_Model)
- **Objective:** This project aims to develop a Computer vision model that can identify the type of clothing from its picture.

- **Dataset:** The dataset used for this project is the Fashion MNIST dataset, which is a dataset of images of clothing items. It consists of 60,000 images for training and 10,000 images for testing.

- **Model Architechture:** The final model used 3 CNN blocks and a single densly connected layer to achieve the task

- **Model Accuracy:** The model achieved an **accuracy of 91%** and a **loss of 0.26** on the test dataset.
