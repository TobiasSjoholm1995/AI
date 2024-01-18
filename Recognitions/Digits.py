# download https://ghostscript.com/releases/gsdnld.html  (restart IDE after)
# pip install tensorflow
# pip install pillow
# pip install opencv-python
# pip install tk

import os
import io
import cv2 
import numpy as np
import tkinter as tk
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers   import BatchNormalization, Flatten, Dropout, Dense, Conv2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils    import to_categorical
from tensorflow.keras.models   import Sequential, save_model, load_model


IMAGE_SIZE_X      = 28
IMAGE_SIZE_Y      = 28
DEBUG_MODE        = False
IMAGE_FILEPATH    = "digit.jpg"
MODIFIED_FILEPATH = "digit_preprocessed.jpg"
MODEL_FILEPATH    = "digits_model.h5"


class DataManager:

   @staticmethod
   def to_black_background(img):
      if img is None:
         return img
      
      treshhold  = 128
      mean_value = np.mean(img)
      return img if mean_value < treshhold else cv2.bitwise_not(img)


   @staticmethod
   def normalize_image(img, count):
      if img  is None:
         return img
      
      image_bands    = 1 # makes it gray based
      normalized_rgb = 255
      return img.reshape((count, IMAGE_SIZE_X, IMAGE_SIZE_Y, image_bands)).astype('float32') / normalized_rgb


   @staticmethod
   def get_mnist_data():
      (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
      train_images = DataManager.normalize_image(train_images, 60_000)
      test_images  = DataManager.normalize_image(test_images, 10_000)
      train_labels = to_categorical(train_labels)
      test_labels  = to_categorical(test_labels)
      return train_images, train_labels, test_images, test_labels


   @staticmethod
   def get_user_image(filepath):
      if not os.path.exists(filepath):
         print(f'File dont exist: {filepath}')
         return None
      
      img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      img = DataManager.to_black_background(img)
      img = cv2.resize(img, (IMAGE_SIZE_X, IMAGE_SIZE_Y))

      if DEBUG_MODE:
         cv2.imwrite(MODIFIED_FILEPATH, img)

      return DataManager.normalize_image(img, 1)


class AI:
   def __init__(self):
      self.model = self.get_model()
      self.train_model()


   def train_model(self):
      if self.is_trained:
         return
      
      train_images, train_labels, test_images, test_labels = DataManager.get_mnist_data()
      self.model.fit(train_images, train_labels, epochs=20, batch_size=64)
      test_loss, test_acc = self.model.evaluate(test_images, test_labels)

      print(f'Test Accuracy: {test_acc}')
      save_model(self.model, MODEL_FILEPATH)


   def get_model(self):
      self.is_trained = os.path.exists(MODEL_FILEPATH)

      if self.is_trained:
         return load_model(MODEL_FILEPATH)

      model = Sequential()
      model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
      model.add(BatchNormalization())
      model.add(Conv2D(32,kernel_size=3,activation='relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
      model.add(BatchNormalization())
      model.add(Dropout(0.4))
      model.add(Conv2D(64,kernel_size=3,activation='relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(64,kernel_size=3,activation='relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
      model.add(BatchNormalization())
      model.add(Dropout(0.4))
      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(BatchNormalization())
      model.add(Dropout(0.4))
      model.add(Dense(10, activation='softmax'))

      model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
      return model


   def predict_image(self):
      img    = DataManager.get_user_image(IMAGE_FILEPATH)
      result = self.model.predict(img, verbose = 0)
      digit  = np.argmax(result)
      conf   = round(result[0][digit] * 100)
      result = f'Prediction: {digit}' + '  ' + f'Confidence: {conf}%'
      return result


class PaintApp:
   def __init__(self, root, predict_digit):
      self.predict_digit = predict_digit
      self.root = root
      self.root.title("Draw Something!")
      self.root.resizable(width=False, height=False)
      
      self.canvas = tk.Canvas(root, bg="white", width=500, height=400)
      self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
      
      self.brush_size = 30
      self.color = "black"
      self.old_x = None
      self.old_y = None

      self.canvas.bind("<B1-Motion>", self.paint)
      self.canvas.bind("<ButtonRelease-1>", self.paint_completed)
      self.canvas.bind("<Button-3>", self.clear_canvas)


   def paint(self, event):
      x, y = event.x, event.y
     
      if self.old_x and self.old_y:
         self.canvas.create_line(self.old_x, self.old_y, x, y, width=self.brush_size, fill=self.color, capstyle=tk.ROUND, smooth=tk.TRUE)

      self.old_x = x
      self.old_y = y


   def paint_completed(self, event):
      self.old_x = None
      self.old_y = None
      self.save_image()


   def clear_canvas(self, pos):
      self.canvas.delete("all")
      self.root.title("Draw Something!")


   def save_image(self):
      self.root.update()
      ps  = self.canvas.postscript(colormode="color")
      img = Image.open(io.BytesIO(ps.encode('utf-8')))
      img.save(IMAGE_FILEPATH, "JPEG")
      self.root.title(self.predict_digit())


def main():
   try:
      root = tk.Tk()
      ai   = AI()
      app  = PaintApp(root, ai.predict_image)

      root.mainloop()
   except Exception as err:
      print("An error occurred:", err)



if __name__ == "__main__":
   main()
