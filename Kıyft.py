#Library
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import matplotlib.pyplot as plt
from glob import glob
import keras

path='./train/'

img=load_img(path+ "dress/0a69db60-c052-4b9a-a90d-e53120d091d5.jpg")

plt.imshow(img)
plt.axis("off")
plt.show()

x=img_to_array(img)
print("Shape=",x.shape)

className= glob(path + '/*')
numberOfClass= len(className)
print("Number of class=",numberOfClass)

#%% CNN MODEL
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.3))
model.add(Dense(numberOfClass)) #output
model.add(Activation("softmax"))


model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

batch_size=32

#%% Data Generation - Train - Test
train_datagen = ImageDataGenerator(rescale=1./255, #RGB'den dolayı
                   shear_range=0.3,#döndürme
                   horizontal_flip=True,#tamamen yançevir
                   zoom_range=0.3,  #zoom ölçeği 
                   validation_split=0.2,
                   rotation_range=20
                   )
train_generator = train_datagen.flow_from_directory(
    path,
    target_size=(x.shape[:2]),
    batch_size = batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset='training')

test_generator = train_datagen.flow_from_directory(
    path,
    target_size=(x.shape[:2]),
    batch_size = batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset='validation')


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,#kalan resimler datageneratörden geliyor
    epochs=200,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size) 
    

#%% Model Evaluation

model.save_weights("deneme.h5") #model kaydetme

#%% Model Evaluation
print(model.history)

#%% model evaluation

plt.figure()
plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(history.history["accuracy"], label = "Train acc")
plt.plot(history.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()
#%% Pred
import numpy as np
import PIL.Image

img = PIL.Image.open("elbise.jpg")
img = img.resize((301,205))
img = np.array(img)
img = img / 255.0
img = img.reshape(1,301,205,3)
result = model.predict(img)

#%% Pred

print(result)
#%% Pred
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  path )

class_names = train_ds.class_names
print(class_names)

image_path="tayt.jpg"

img = keras.preprocessing.image.load_img(
    image_path, target_size=(301,205)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

