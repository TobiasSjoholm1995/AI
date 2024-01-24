import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model


SKIP_TRAINING   = False
DATA_SIZE_DIM_1 = 10
DATA_SIZE_DIM_2 = 20
DATA_SHAPE      = (DATA_SIZE_DIM_1, DATA_SIZE_DIM_2)
INPUT_DIM       = 100
BATCH_SIZE      = 64
EPOCHS          = 15_000
DIS_OVERFITTING = 0.95
GEN_PROGRESSION = 0.1
DIS_EPOCH_STOP  = 14_500
GAN_EPOCH_START = 300


class Network:

   def __init__(self):
      self.save_folder  = "Models"
      self.name_prefix  = "matrix"
      self.gen_filename = self.name_prefix + "_" + 'generator.h5'
      self.dis_filename = self.name_prefix + "_" + 'discriminator.h5'
      self.gan_filename = self.name_prefix + "_" + 'gan.h5'

      self.gen = self.build_generator()
      self.dis = self.build_discriminator()
      self.gan = self.build_gan()


   def save(self, epoch):
      folder = self.save_folder

      if not os.path.exists(folder):
         os.makedirs(folder)

      prefix = "Epoch_" + str(epoch) +  '_'
      self.gen.save(os.path.join(folder, prefix + self.gen_filename))
      self.dis.save(os.path.join(folder, prefix + self.dis_filename))
      self.gan.save(os.path.join(folder, prefix + self.gan_filename))


   def build_generator(self):
      if os.path.exists(self.gen_filename):
         return load_model(self.gen_filename)

      model = models.Sequential()
      model = models.Sequential()
      model.add(layers.Dense(64, input_dim=INPUT_DIM, activation='relu'))
      model.add(layers.Dense(128, activation='relu'))
      model.add(layers.Dense(DATA_SIZE_DIM_1 * DATA_SIZE_DIM_2, activation='sigmoid'))  
      model.add(layers.Reshape(DATA_SHAPE))
      model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
      return model


   def build_discriminator(self):
      if os.path.exists(self.dis_filename):
         return load_model(self.dis_filename)

      model = models.Sequential()
      model.add(layers.Flatten(input_shape=DATA_SHAPE))  
      model.add(layers.Dense(128, activation='relu'))
      model.add(layers.Dense(64, activation='relu'))
      model.add(layers.Dense(1, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.006, 0.5), metrics=['accuracy'])
      return model


   def build_gan(self):
      if os.path.exists(self.gan_filename):
         return load_model(self.gan_filename)

      self.dis.trainable = False
      model = models.Sequential()
      model.add(self.gen)
      model.add(self.dis)
      model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
      return model


class DataManager:
   data = None

   @staticmethod
   def generate_valid_data(rows, cols):
      matrix = np.zeros((rows, cols), dtype=int)
      r1, c1 = random.randint(0, rows - 1), random.randint(0, cols - 1)
      r2, c2 = r1, c1
      
      while abs(r1 - r2) <= 1:
         r2 = random.randint(0, rows - 1)
      while abs(c1 - c2) <= 1:
         c2 = random.randint(0, cols - 1)
      
      matrix[r1, :] = 1
      matrix[r2, :] = 1
      matrix[min(r1,r2):max(r2,r1), c1] = 1
      matrix[min(r1,r2):max(r2,r1), c2] = 1

      return matrix
      

   @classmethod
   def get_training_data(cls, batch_size=0):
      if cls.data is None:
         count    = 100_000
         cls.data = np.array([DataManager.generate_valid_data(DATA_SIZE_DIM_1, DATA_SIZE_DIM_2) for _ in range(100_000)]) 
         cls.data = cls.data.reshape((count,) + DATA_SHAPE)

      return cls.data[np.random.randint(0, cls.data.shape[0], batch_size)] if batch_size > 0 else cls.data
      

   @staticmethod
   def generate_data(network, count):
      noise  = np.random.normal(0, 1, (count, INPUT_DIM))
      data   = network.gen.predict(noise, verbose=0)
      scores = network.dis.predict(data, verbose=0).flatten()
      scores = [int(round(100 * s)) for s in scores]
      data   = data.reshape((count,) + DATA_SHAPE)
      return data, scores


class Display:

   @staticmethod
   def progression(network, epoch, count = 1):
      gen_data, gen_scores = DataManager.generate_data(network, count)
      gen_titles = [f'Gen_Epoch_{epoch}_Score_{s}:' for s in gen_scores]

      real_data  = DataManager.get_training_data(count).reshape((count,) + DATA_SHAPE)
      dis_data   = network.dis.predict(real_data, verbose=0)[0]
      avg_score  = int(round(sum(100 * dis_data) / len(dis_data)))
      real_title = f'Real_Epoch_{epoch}_Score_{avg_score}:' 

      print(real_title)
      Display.data(gen_data, gen_titles)


   @staticmethod
   def result(network, count = 1):
      gen_data, gen_scores = DataManager.generate_data(network, count)
      gen_names = [f'Gen_Result_{i}_Score_{s}:' for i, s in enumerate(gen_scores)]
      Display.data(gen_data, gen_names)
 


   @staticmethod
   def data(data, titles):   
      for i in range(len(data)):
         print(titles[i])
         arrays = data[i]

         for arr in arrays:
            for value in arr:
               print(int(round(value)), end=' ')
            print()
         print()


class Train:

   @staticmethod
   def dis(network, epoch):
      half_batch  = int(BATCH_SIZE / 2)
      noise       = np.random.normal(0, 1, (half_batch, INPUT_DIM))
      real_data   = DataManager.get_training_data(half_batch)
      gen_data    = network.gen.predict(noise, verbose=0)
      real_labels = np.ones((half_batch, 1)) * DIS_OVERFITTING
      gen_labels  = np.zeros((half_batch, 1)) * (GEN_PROGRESSION * epoch / EPOCHS)
      x           = np.vstack((real_data, gen_data))
      y           = np.vstack((real_labels, gen_labels))
      network.dis.train_on_batch(x, y)  


   @staticmethod
   def gan(network):
      noise   = np.random.normal(0, 1, (BATCH_SIZE, INPUT_DIM))
      valid_y = np.ones((BATCH_SIZE, 1))
      network.gan.train_on_batch(noise, valid_y) 


   @staticmethod
   def network(network):
      if SKIP_TRAINING: 
         return

      print(f'Total epoch count: {EPOCHS}\n')

      for epoch in range(EPOCHS):
         if epoch < DIS_EPOCH_STOP:
            Train.dis(network, epoch)

         if epoch > GAN_EPOCH_START:
            Train.gan(network)

         if epoch % 100 == 0:
            Display.progression(network, epoch)

         if epoch % 1000 == 0:
            network.save(epoch)


def main():
   network = Network()
   
   Train.network(network)
   network.save(EPOCHS)
   Display.result(network, 10)


if __name__ == "__main__":
   try:
      print()
      main()
      print()
   except Exception as err:
      print(err)
      print()
