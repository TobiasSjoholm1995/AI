## Neural Network Experiments

### Overview
This repository contains Python scripts that explore various types of neural network architectures to solve different problems, including binary classification, multi-class classification and hand-writing recognition. The purpose of these experiments is to evaluate the performance of different neural network models on diverse tasks and gain insights into their strengths and limitations.

### Project Structure
The repository is organized as follows:

- Classifications: This directory contains Python scripts that focus on solving classifications problems, such as Binary and Multi-Class classifications. 

- Recognition: Delves into the realm of neural network models designed for the identification of handwritten characters. Leveraging the NMIST dataset for training data. The script employs the TKinter library to create a graphical user interface (GUI). This interface enables users to draw digits and the neural network tries to predict the digit. The GUI not only displays the prediction but also conveys the confidence associated with the model's recognition.

- Info: Contains information about activation functions and loss functions that can be used as a lookup when needed in the future.

### Requirements
To run the scripts in this repository, ensure that you have the following dependencies installed:

- Python (version 3.x)
- TensorFlow (version 2.x)
- NumPy
- TKinter
- GhostrScript (for postscripts to covnert ps to JPG images)
- OpenCV
- PIL

Make sure to download https://ghostscript.com/releases/gsdnld.html  (restart IDE after)
You can install the required Python packages using the following commands:
- pip install tensorflow
- pip install pillow
- pip install opencv-python
- pip install tk
- pip install PIL


## License
This project is licensed under the MIT License.
