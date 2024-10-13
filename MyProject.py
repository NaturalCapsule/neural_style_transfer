import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

person_image = load_img('person.jpg')
style_image = load_img('starry_night.jpg')

stylized_image = model(tf.constant(person_image), tf.constant(style_image))[0]

cv2.imwrite('output_image.jpg', cv2.cvtColor(np.squeeze(stylized_image) * 255, cv2.COLOR_BGR2RGB))