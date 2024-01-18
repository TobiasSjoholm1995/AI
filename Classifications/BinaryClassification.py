import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class Agent:
   def __init__(self):
      self.model = self.get_model()


   def get_model(self):
      #dont need this heavy NN, but just to show that u can use dropout if you want...
      model = Sequential()
      model.add(Dense(256, activation='relu', input_shape=(2,)))
      model.add(Dropout(0.5))
      model.add(Dense(128, activation='relu'))
      model.add(Dense(64,  activation='relu'))
      model.add(Dense(64,  activation='relu'))
      model.add(Dropout(0.5))
      model.add(Dense(1,   activation='sigmoid'))
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      return model


   def train(self, input, output):
      self.model.fit(input, output, epochs=20, batch_size=64)


class Env:

   def get_data(self, count, print=False):
      x      = np.random.randint(-10, 11, size=count)
      y      = np.random.randint(0, 101,  size=count)
      input  = np.column_stack((x, y))
      output = np.array([self.get_output(p) for p in input]) 
      data   = (input, output)

      if print:
         self.print_data(data)
         
      return data
    
    
   def get_output(self, input):
      x = input[0]
      y = input[1]
      r = 1 if math.pow(x, 2) > y else 0
      return r
        
        
   def print_data(self, data):
      input, output = data

      for i, o in zip(input, output):
         print(str(i) + " --> " + str(o))


   def verify(self, agent):
      data   = self.get_data(7)
      input  = data[0]
      output = data[1]
      result = agent.model.predict(input)
  
      for i,o,r in zip(input, output, result):
         print(f"{str(i).ljust(9)} --> {'{:.2f}'.format(r[0])} ({o})")

    
env   = Env()
agent = Agent()
data  = env.get_data(100_000)

agent.train(*data)
env.verify(agent)

