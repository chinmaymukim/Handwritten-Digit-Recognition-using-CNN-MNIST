# Handwritten-Digit-Recognition-using-CNN-MNIST
📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST. The objective is to build an end-to-end deep learning pipeline covering data preprocessing, model design, training, validation, and evaluation.

The model is implemented using Keras with TensorFlow backend.

📊 Dataset Details

The MNIST dataset consists of grayscale images of handwritten digits (0–9).

Training Samples: 60,000

Testing Samples: 10,000

Image Size: 28 × 28 pixels

Number of Classes: 10

Dataset loading:

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


Initial dataset shapes:

X_train.shape → (60000, 28, 28)

y_train.shape → (60000,)

X_test.shape → (10000, 28, 28)

y_test.shape → (10000,)

🔎 Data Preprocessing

The following preprocessing steps were applied:

1️⃣ Normalization

Pixel values were converted to float32 and scaled to the range [0,1]:

X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

2️⃣ Reshaping

Expanded image dimensions to make them compatible with CNN input format:

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

3️⃣ One-Hot Encoding

Converted class labels into categorical format:

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

🧠 Model Architecture

The model is built using the Sequential API in Keras.

Architecture Details:

Conv2D

32 filters

Kernel size: (3,3)

Activation: ReLU

MaxPooling2D

Pool size: (2,2)

Conv2D

64 filters

Kernel size: (3,3)

Activation: ReLU

MaxPooling2D

Pool size: (2,2)

Flatten

Dropout

Rate: 0.25

Dense (Output Layer)

10 neurons

Activation: Softmax

Model Summary

Total Parameters: 34,826

Trainable Parameters: 34,826

Non-Trainable Parameters: 0

⚙️ Model Compilation

The model was compiled with:

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

model.compile(
    optimizer='adam',
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

⏱️ Callbacks Used
EarlyStopping

Monitor: val_acc

Min Delta: 0.01

Patience: 4

ModelCheckpoint

File Path: ./bestmodel.h5

Monitor: val_acc

Save Best Only: True

🚀 Model Training

Training Configuration:

Epochs: 5

Validation Split: 0.3

his = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.3
)

Final Training Metrics:

Training Accuracy: ~95.99%

Validation Accuracy: 97.11%

Validation Loss: 0.0993

📈 Model Evaluation

The saved model was loaded and evaluated on the test dataset:

model_S = keras.models.load_model('my_bestmodel.h5')
score = model_S.evaluate(X_test, y_test)

🎯 Test Accuracy:

97.50%

My model accuracy is 0.9750000238418579

🛠️ Libraries Used

NumPy

Matplotlib

Keras

TensorFlow (backend)

📌 Key Highlights

End-to-end CNN implementation

Data normalization and reshaping

One-hot encoding for multi-class classification

Regularization using Dropout

Use of EarlyStopping and ModelCheckpoint

Achieved 97.5% accuracy on test data

📬 Conclusion

This project demonstrates a complete deep learning workflow for image classification using Convolutional Neural Networks. It covers data preprocessing, model design, training, validation, and evaluation using the MNIST benchmark dataset.
