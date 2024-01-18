import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class Agent:
   def __init__(self):
      self.model = self.get_model()


   def get_model(self):
      model = Sequential()
      model.add(Dense(8, activation='relu', input_shape=(2,)))
      model.add(Dense(4, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      return model


   def train(self, input, output):
      self.model.fit(input, output, epochs=10, batch_size=64)


class Env:

   def get_data(self, count, print=False):
      x      = np.random.randint(-100, 101, size=count)
      y      = np.random.randint(-100, 101, size=count)
      input  = np.column_stack((x, y))
      output = np.array([self.get_output(p) for p in input]) 
      data   = (input, output)

      if print:
         self.print_data(data)
         
      return data
    
    
   def get_output(self, input):
      x = input[0]
      y = input[1]

      if x == 0 and y == 0:
         return [0,0,0,0]
      
      if x >= 0 and y >= 0:
         return [1, 0, 0, 0]
      
      if x > 0 and y < 0:
         return [0, 1, 0, 0]
      
      if x <= 0 and y <= 0:
         return [0, 0, 1, 0]
      
      if x < 0 and y > 0:
         return [0, 0, 0, 1]
        
        
   def print_data(self, data):
      input, output = data

      for i, o in zip(input, output):
         print(str(i) + " --> " + str(o))


   def verify(self, agent):
      data   = self.get_data(7)
      input  = data[0]
      output = data[1]
      result = agent.model.predict(input)

      print('--input--      --predicted--            --output--')
      for i,o,r in zip(input, output, result):
         formatted_result = ", ".join("{:.2f}".format(number) for number in r)
         print(f"{str(i).ljust(10)} --> [{formatted_result}]  {o}")

    
env   = Env()
agent = Agent()
data  = env.get_data(100_000)

agent.train(*data)
env.verify(agent)


