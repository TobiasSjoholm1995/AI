## Neural Network Experiments

### Overview
This repository contains Python scripts that explore various types of neural network architectures to solve different problems, including binary classification, multi-class classification and hand-writing recognition. The purpose of these experiments is to evaluate the performance of different neural network models on diverse tasks and gain insights into their strengths and limitations.

### Project Structure
The repository is organized as follows:

- Classifications: This directory contains Python scripts that focus on solving classifications problems, such as Binary and Multi-Class classifications. 
- Recognitions: This section explores neural network models for the recognition of handwritten characters, using NMIST dataset. The script also utilize TKinter library for the graphic user interface (GUI), so the user has the cababilities to draw digits so the neural network can guess the digit. Both the prediction and the confidence are presented in the GUI.
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

You can install the required packages using the following command:

Download https://ghostscript.com/releases/gsdnld.html  (restart IDE after)
pip install tensorflow
pip install pillow
pip install opencv-python
pip install tk
pip install PIL


## License
This project is licensed under the MIT License.
