import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model


IMAGE_SIZE    = 28
IMAGE_SHAPE   = (IMAGE_SIZE, IMAGE_SIZE, 1)
INPUT_DIM     = 100
BATCH_SIZE    = 64
EPOCHS        = 30_000
SKIP_TRAINING = False
GAN_FILEPATH  = 'model_gan.h5'
GEN_FILEPATH  = 'model_generator.h5'
DIS_FILEPATH  = 'model_discriminator.h5'


def build_generator():
   if os.path.exists(GEN_FILEPATH):
      return load_model(GEN_FILEPATH)
   
   model = models.Sequential()
   model.add(layers.Dense(7 * 7 * 128, input_dim=INPUT_DIM))
   model.add(layers.Reshape((7, 7, 128)))
   model.add(layers.UpSampling2D())
   model.add(layers.Conv2D(128, kernel_size=3, padding='same'))
   model.add(layers.BatchNormalization(momentum=0.8))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.UpSampling2D())
   model.add(layers.Conv2D(64, kernel_size=3, padding='same'))
   model.add(layers.BatchNormalization(momentum=0.8))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2D(1, kernel_size=3, padding='same'))
   model.add(layers.Activation('sigmoid'))
   model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
   return model


def build_discriminator():
   if os.path.exists(DIS_FILEPATH):
      return load_model(DIS_FILEPATH)

   model = models.Sequential()
   model.add(layers.Conv2D(64, kernel_size=3, strides=2, input_shape=IMAGE_SHAPE, padding='same'))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Dropout(0.25))
   model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding='same'))
   model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
   model.add(layers.BatchNormalization(momentum=0.8))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Dropout(0.25))
   model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
   model.add(layers.BatchNormalization(momentum=0.8))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Dropout(0.25))
   model.add(layers.Flatten())
   model.add(layers.Dense(1, activation='sigmoid'))
   model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
   return model


def build_gan(generator, discriminator):
   discriminator.trainable = False
   if os.path.exists(GAN_FILEPATH):
      return load_model(GAN_FILEPATH)

   model = models.Sequential()
   model.add(generator)
   model.add(discriminator)
   model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
   return model


def get_nmist_data():
   (images, _), (_, _) = tf.keras.datasets.mnist.load_data()
   count   = 60000
   images  = images.reshape((count, IMAGE_SIZE, IMAGE_SIZE, 1)).astype('float32') / 255.0
   return images


def train(generator, discriminator, gan, train_images):
   if SKIP_TRAINING: 
      return
   
   half_batch = int(BATCH_SIZE / 2)
   print(f'Total epoch count: {EPOCHS}\n')

   for epoch in range(EPOCHS): 
      indexes     = np.random.randint(0, train_images.shape[0], half_batch)
      noise       = np.random.normal(0, 1, (half_batch, INPUT_DIM))
      real_images = train_images[indexes]
      gen_images  = generator.predict(noise, verbose=0)
      real_labels = np.ones((half_batch, 1))
      gen_labels  = np.zeros((half_batch, 1))
      x           = np.vstack((real_images, gen_images))
      y           = np.vstack((real_labels, gen_labels))
      _, d_acc    = discriminator.train_on_batch(x, y)  

      noise       = np.random.normal(0, 1, (BATCH_SIZE, INPUT_DIM))
      valid_y     = np.ones((BATCH_SIZE, 1))
      g_loss      = gan.train_on_batch(noise, valid_y) 

      if epoch % 100 == 0:
         show_progression(generator, epoch)

      if epoch % 1000 == 0:
         save_models(generator, discriminator, gan)


def show_progression(generator, epoch):
   name  = f'generated_image_epoch_{epoch}.jpg'
   image = generate_images(generator, 1)[0]
   
   save_image(image, name)
   print(f'Saved {name}')


def show_result(generator):
   count  = 10
   images = generate_images(generator, count)
   
   for i in range(count):
      save_image(images[i], f'generated_image_result_{i+1}.jpg')


def generate_images(generator, count):
   noise  = np.random.normal(0, 1, (count, INPUT_DIM))
   images = generator.predict(noise, verbose=0)
   images = images.reshape((count, IMAGE_SIZE, IMAGE_SIZE))
   return images


def save_image(image, filename):
   plt.imshow(image, cmap='gray')
   plt.axis('off')
   plt.savefig(filename, format='jpeg', bbox_inches='tight', pad_inches=0)


def save_models(gen, dis, gan):
   gen.save(GEN_FILEPATH)
   dis.save(DIS_FILEPATH)
   gan.save(GAN_FILEPATH)


def main():
   data          = get_nmist_data()
   generator     = build_generator()
   discriminator = build_discriminator()
   gan           = build_gan(generator, discriminator)
   
   train(generator, discriminator, gan, data)
   save_models(generator, discriminator, gan)
   show_result(generator)


if __name__ == "__main__":
   try:
      print()
      main()
      print()
   except Exception as err:
      print(err)
      print()
