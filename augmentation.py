from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = Image.open('example.png')

img = img.convert('RGB')

img_array = np.array(img)

img_array = img_array.reshape((1,) + img_array.shape)

i = 0
for batch in datagen.flow(img_array, batch_size=1,
                          save_to_dir='.',
                          save_prefix='parshuram',
                          save_format='jpeg'):

    i += 1
    if i > 10: 
        break
