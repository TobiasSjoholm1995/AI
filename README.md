## Neural Network Experiments

### Overview
This repository contains Python scripts that explore various types of neural network architectures to solve different problems, including classification, regression, generation and recognition. The purpose of these experiments is to evaluate the performance of different neural network models on diverse tasks and gain insights into their strengths and limitations.


### Project Structure
The repository is organized as follows:

- __Classifications__: This directory contains Python scripts that focus on solving classifications problems, such as Binary and Multi-Class classifications.

- __Generations__: Entails the implementation of a Generative Adversarial Network (GAN), encompassing the definition of the generator, discriminator, and GAN neural network models. The training process revolves around generating synthetic images and fine-tuning the weights of the neural network models. Additionally, the project incorporates features for visualizing and saving the synthetic generated images.

- **Recognitions**: Delves into the realm of neural network models designed for the identification of handwritten characters. Leveraging the NMIST dataset for training data. The script employs the TKinter library to create a graphical user interface (GUI). This interface enables users to draw digits and the neural network tries to predict the digit. The GUI not only displays the prediction but also conveys the confidence associated with the model's recognition.

- **Regressions**: Conducts linear and polynomial regressions through the utilization of neural networks.


### Screenshot Preview

Handwriting recognition through an artificial neural network:

![image](https://github.com/TobiasSjoholm1995/AI/assets/43572826/404906f8-f405-44ce-b03c-e0379122b17b)


### Requirements
To run the scripts in this repository, ensure that you have the following dependencies installed:

- Python:  https://www.python.org/downloads/
- GhostScript:  https://ghostscript.com/releases/gsdnld.html 
- TensorFlow:  pip install tensorflow 
- NumPy:  pip install np
- TKinter:  pip install tk
- OpenCV:  pip install opencv-python
- PIL:  pip install pillow


## License
This project is licensed under the MIT License.
