import cv2
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
model=tensorflow.keras.models.load_model('model_test_v3.h5')
test_image=image.load_img('t2.png',target_size=(1280,720))
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)

cv2.imshow("video1",test_image)
print(test_image)

if result[0][0] == 1:
    print("cember")
elif result[0][1] == 1:
    print("other")
elif result[0][2] == 1:
    print("cember")
elif result[0][3] == 1:
    print("other")
