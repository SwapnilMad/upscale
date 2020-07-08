from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim


model = load_model('generator_1.h5')

image = Image.open('dandelion.jpg')
image = image.resize((640, 400))
image.show()
img1 = np.uint8(image)
im = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
image = np.uint8(im)

for i in range(1):
    image = (image - 127.5) / 127.5
    pred = model.predict(np.array([image]))
    pred = (pred[0] * 127.5) + 127.5
    image = np.uint8(pred)
    img2 = image

psnr = tf.image.psnr(img1, img2, 255)
print(psnr)
ssi = ssim(img1, img2, data_range=img1.max() - img2.min(), multichannel=True)
print('psnr:', psnr, 'ssim:', ssi)

im = Image.fromarray(image)

im.show()
