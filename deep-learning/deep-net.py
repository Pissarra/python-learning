import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files
import numpy as np
from tensorflow.python import metrics
##--- Download Dataset
#Dowload the file
# !wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

#Decompress the folder
local_zip = './cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

## -- Variable Definition

# declare the location of out training and validation files
base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

#cat and dog folders for training
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

#cat and dog folders for validation
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Quantidade de imagens
print(len(os.listdir(train_cats_dir)))
print(len(os.listdir(train_dogs_dir)))
print(len(os.listdir(validation_cats_dir)))
print(len(os.listdir(validation_dogs_dir)))

import tensorflow as tf

model = tf.keras.models.Sequential([
    # since Conv2D is the first layer of the neural network, we should also specify the size of the input
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    # apply pooling
    tf.keras.layers.MaxPooling2D(2,2),
    # and repeat the process
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #flatten the result to feed it to the dense layer
    tf.keras.layers.Flatten(),
    # and define 512 neurons for processing the output coming by the previous layers
    tf.keras.layers.Dense(512, activation='relu'),
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model.summary()



model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

from tensorflow.python.ops.batch_ops import batch
#pré-processamento das imagens


#we rescale all our images with the rescale parameter
train_datagen = ImageDataGenerator(rescale = 1.0/255)
test_datagen = ImageDataGenerator(rescale = 1.0/255)

#we use flow_from_directory to create a generator for training
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150))

#we use flow_from_directory to create a generator for validation
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150,150))

#Treinamento do modelo
history = model.fit(
    train_generator, #pass in the training generator
    steps_per_epoch=100,
    epochs=15,
    validation_data=validation_generator, #pass in the validation generator
    validation_steps=50,
    verbose=2
)

# get the metrics from history
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# plot accuracy with matplotlib
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Accuracy in training and validation')
plt.figure()

# plot loss with matplotlib
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Loss in training and validation')




# from keras.preprocessing import image
from keras.utils import load_img, img_to_array

uploaded = files.upload()

for fn in uploaded.keys():

    # prediction on the uploaded image
    path = '/content/' + fn  # load the image on Colab
    img = load_img(path, target_size=(150, 150))  # let's use load_img to scale it

    # scaling process
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    # flatten the output
    images = np.vstack([x])

    # prediction!
    classes = model.predict(images, batch_size=10)

    print(classes[0])

    if classes[0] > 0.5:
        print(fn + " é um cachorro!")
    else:
        print(fn + " é um gato!")