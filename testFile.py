import os
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

model = load_model('test.h5')

# classifications tallied up
output1 = 0
output0 = 0

for filename in os.listdir("G:\\Images\\mountainsTest\\positive\\"):
    test_image = image.load_img(str('G:\\Images\\mountainsTest\\positive\\' + str(filename)), target_size = (256, 256, 3))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict(test_image)
    if result[0][0] == 1:
        output1 += 1
    else:
        output0 += 1

print("Classified as Mountain:", output1)
print("Classified as not a Mountain:", output0)

