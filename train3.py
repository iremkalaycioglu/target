import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

os.environ['KMP_DUPLICATE_LIB_OK']='True'

######## Veri ön hazılık (Basit)#######
train_datagen = ImageDataGenerator(rescale=1 / 255, horizontal_flip=True, rotation_range=75)
test_datagen = ImageDataGenerator(rescale=1 / 255)
batch_size = 16
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(1280, 720),
    batch_size=batch_size,
    classes=['cember', 'other' ],
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(1280, 720),
    batch_size=batch_size,
    classes=['cember', 'other' ],
    class_mode='categorical')

## CNN sıralı model oluşturma
model = Sequential([
    # input layer
    Convolution2D(32, (3, 3), activation='relu', input_shape=(1280, 720, 3)),
    MaxPooling2D(2, 2),  ## gausslama mantığı
    # hidden layer 1
    Convolution2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # hidden layer 2
    Convolution2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # hidden layer 3
    Convolution2D(8, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),  ## matematiksel olarak model eğitim bilgilerini dizi haline çevirme

    # Multiconnectir full Connector
    Dense(128, activation='relu'),

    # Output Layer
    Dense(2, activation='softmax')
    # 2 den daha fazla sınıflarda  relu yerine softmax aktivasyon fonksiyonu tercih edilir

])

model.summary()

# Model Optimizasyonu
opt = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])  # accuracy

epoch = 100
tarihce = model.fit(
    train_generator,
    epochs=epoch,
    steps_per_epoch=10,
    shuffle=True,
    validation_data=test_generator,
    validation_steps=20,
    verbose=1)

model.save('model_test_v6.h5')
