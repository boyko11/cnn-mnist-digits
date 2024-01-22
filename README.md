# CNN Refresher over MNIST with PyTorch

## Overview
This project is a quick refresher on building and training Convolutional Neural Networks (CNNs) using Python and PyTorch. It demonstrates a simple CNN architecture applied to the famous MNIST dataset for handwritten digit recognition.

## Features
- Implemented in Python using PyTorch, a popular deep learning framework.
- Utilizes the MNIST dataset, loaded directly from PyTorch's dataset utilities.
- Uses Poetry for dependency management and project setup.
- Flexible architecture that allows for easy adjustments and experimentation.

## Getting Started
To run the project, ensure you have Poetry installed and set up on your system. Then, follow these steps:

1. Clone the repository:

   ```bash
   git clone git@github.com:boyko11/cnn-mnist-digits.git
   ```

2. Navigate to the project directory:

   ```bash
    cd cnn-mnist-digits
    ```
   
3. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

4. Run the pipeline:

   ```bash
    poetry run python pipeline.py
   ```


## Customizing the CNN Architecture
You could adjust the instantiation of the CNN class in `pipeline.py` to experiment with different parameters.   
The following is an example of a CNN with 2 convolutional layers, 2 max pooling layers, and 2 fully connected layers:
```python
net = CNN(
 in_channels=1,
 img_width=28,
 img_height=28,
 conv1_out_channels=8,
 conv1_kernel_size=5,
 max_pool1_kernel_size=2,
 conv2_out_channels=18,
 conv2_kernel_size=5,
 max_pool2_kernel_size=2,
 fc1_out_features=120,
 fc2_out_features=84,
 num_classes=10
)
