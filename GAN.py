from tensorflow.keras.layers import Dropout, Conv2D, Dense, LeakyReLU, Conv2DTranspose, Flatten
from tensorflow.keras.models import Sequential
import os.path
import pickle
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import numpy as np
from PIL import Image
import os
from tensorflow import GradientTape
import tensorflow as tf


class GAN:
    def __init__(self, batch_size, sample_interval):
        self.batch_size = batch_size
        self.img_rows = 40
        self.img_cols = 40
        self.channels = 3
        self.sample_interval = sample_interval
        dir = os.path.dirname(os.path.abspath(__file__))
        self.pickle_dir = os.path.join(dir, "images_pickle")
        self.count = 0
        self.min_loss = 100.0
        self.discriminator = self.build_discriminator()
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.generator = self.build_generator()

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def build_generator(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(1, 1), input_shape=(None, None, self.channels), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(16, (5, 5), strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(8, (5, 5), strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', activation='tanh'))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(16, 3, strides=1, input_shape=(self.img_rows, self.img_cols, self.channels), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(16, 3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, 3, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, 3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, 3, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, 3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='sigmoid'))

        return model

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def train_step(self, images, low_res):

        with GradientTape() as gen_tape, GradientTape() as disc_tape:
            generated_images = self.generator(low_res, training=True)
            mse = MeanSquaredError()
            loss = mse(images, generated_images)
            if loss < self.min_loss and self.count > 200:
                gan_name = 'generator.h5'
                self.generator.save(gan_name)
                self.discriminator.save('discriminator.h5')
                self.min_loss = loss

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output) + loss
            disc_loss = self.discriminator_loss(real_output, fake_output) + loss
            if self.count % 100 == 0:
                print(self.count, self.min_loss, gen_loss, disc_loss)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self):
        temp = []
        reduced = []
        for f in os.listdir(self.pickle_dir):
            p = os.path.join(self.pickle_dir, f)
            X_train = pickle.load(open(p, "rb"))
            X_train = X_train['x_train']

            for image in X_train:
                temp.append(image)
                im = Image.fromarray(image)
                im = im.resize((int(im.size[0] / 2), int(im.size[1] / 2)))
                reduced.append(np.uint8(im))

        X_train = np.array(temp)

        reduced = np.array(reduced)
        self.og = X_train
        self.temp = reduced
        while True:
            self.count = self.count + 1
            idx = np.random.randint(0, len(X_train), self.batch_size)
            imgs = X_train[idx]
            imgs = (imgs - 127.5) / 127.5
            noise = reduced[idx]

            noise = (noise - 127.5) / 127.5
            self.train_step(imgs, noise)

            if self.count % self.sample_interval == 0:
                self.sample_images()

    def sample_images(self):
        idx = np.random.randint(0, len(self.temp), 2)
        noise = self.temp[idx]
        noise = (noise - 127.5) / 127.5
        og = np.array(np.uint8(self.og))
        orig = og[idx]
        gen_imgs = self.generator.predict(noise)

        im = (gen_imgs[0] * 127.5) + 127.5
        im = np.uint8(im)
        im = Image.fromarray(im)
        im.save("images/%d.png" % self.count)
        epoch = self.count + 1
        im = Image.fromarray(orig[0])
        im.save("images/%d.png" % epoch)


if __name__ == '__main__':
    gan = GAN(batch_size=10, sample_interval=1000)
    gan.train()
