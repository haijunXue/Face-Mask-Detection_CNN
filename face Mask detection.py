"""
from zipfile import ZipFile
dataset = 'C:/Users/Haijun/PycharmProjects/Pytorch_Übung/face-mask-detection.zip'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('The dataset is extracted')
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

with_mask_files = os.listdir('C:/Users/Haijun/PycharmProjects/Pytorch_Übung/pyTorch_Project/datam/with_mask')
print(with_mask_files[0:5])
print(with_mask_files[-5:])
without_mask_files = os.listdir('C:/Users/Haijun/PycharmProjects/Pytorch_Übung/pyTorch_Project/datam/without_mask')
print(without_mask_files[0:5])
print(without_mask_files[-5:])
print('Number of with image masks', len(with_mask_files))
print('Number of without image masks', len(without_mask_files))

""" 
create lable for the two classes
with mask -->1
without maks -->0
"""

with_mask_labels = [1]*len(with_mask_files)
without_mask_labels = [0]*len(without_mask_files)
print(with_mask_labels[0:5])
print(without_mask_labels[0:5])
labels = with_mask_labels + without_mask_labels
print(len(labels))
print(labels[0:5])
print(labels[-5:])

# display with mask image
img = mping.imread('C:/Users/Haijun/PycharmProjects/Pytorch_Übung/pyTorch_Project/datam/with_mask/with_mask_1.jpg')
imgplot=plt.imshow(img)
plt.show()

img = mping.imread('C:/Users/Haijun/PycharmProjects/Pytorch_Übung/pyTorch_Project/datam/without_mask/without_mask_300.jpg')
imgplot=plt.imshow(img)
plt.show()

# Image processing
with_mask_path = 'C:/Users/Haijun/PycharmProjects/Pytorch_Übung/pyTorch_Project/datam/with_mask'
data = []
for img_file in with_mask_files:
    image = Image.open(os.path.join(with_mask_path, img_file))
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

without_mask_path = 'C:/Users/Haijun/PycharmProjects/Pytorch_Übung/pyTorch_Project/datam/without_mask'
for img_file in without_mask_files:
    image = Image.open(os.path.join(without_mask_path, img_file))
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)
""" 
type(data)
len(data)
print(data[0])
"""

X = np.array(data)
Y = np.array(labels)
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.2, random_state=2 )

X_train_scaled = X_train/255
X_test_scaled = X_test/255

"""
 buliding a convolutional neural network
"""
number_of_classes = 2
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128, 128, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(number_of_classes, activation='sigmoid'))

#compile the neural network
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['acc'])

#training the neural network
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)



# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#model evaluation
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('test Accuracy = ', accuracy )


# Path to the input image
input_image_path =  r'C:\Users\Haijun\PycharmProjects\Pytorch_Übung\pyTorch_Project\datam\without_mask\without_mask_997.jpg'


# Load the image
input_image = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if input_image is None:
    print("Error: Failed to load the image. Please check the file path and file integrity.")
else:
    # Resize the image to the required size for the model (128x128)
    input_image_resized = cv2.resize(input_image, (128, 128))

    # Normalize the image by scaling pixel values to the range [0, 1]
    input_image_scaled = input_image_resized / 255.0

    # Reshape the image to match the input shape expected by the model: (1, 128, 128, 3)
    input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

    # Perform the prediction using the trained model
    input_prediction = model.predict(input_image_reshaped)

    # Output the raw prediction probabilities
    print("Raw prediction output:", input_prediction)

    # Get the index of the highest probability, which corresponds to the predicted class
    input_pred_label = np.argmax(input_prediction)

    # Print the predicted label (0 for 'no mask', 1 for 'mask')
    print("Predicted label:", input_pred_label)

    # Interpret and print the result based on the predicted label
    if input_pred_label == 1:
        print('The person in the image is wearing a mask.')
    else:
        print('The person in the image is not wearing a mask.')