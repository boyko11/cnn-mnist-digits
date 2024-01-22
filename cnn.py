import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,
                 in_channels=1,
                 img_width=28,
                 img_height=28,
                 conv1_out_channels=8,
                 conv1_kernel_size=5,
                 conv1_kernel_padding=0,
                 conv1_kernel_stride=1,
                 max_pool1_kernel_size=2,
                 max_pool1_kernel_stride=2,
                 conv2_out_channels=18,
                 conv2_kernel_size=5,
                 conv2_kernel_padding=0,
                 conv2_kernel_stride=1,
                 max_pool2_kernel_size=2,
                 max_pool2_kernel_stride=2,
                 fc1_out_features=120,
                 fc2_out_features=84,
                 num_classes=10):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, conv1_out_channels, conv1_kernel_size)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, conv2_kernel_size)

        # Calculate output image dimensions after convolution and max pooling
        conv1_img_width_out = CNN.__img_dim_after_convolution(img_width, conv1_kernel_size, conv1_kernel_padding,
                                                               conv1_kernel_stride)
        conv1_img_height_out = CNN.__img_dim_after_convolution(img_height, conv1_kernel_size, conv1_kernel_padding,
                                                                conv1_kernel_stride)
        max_pool1_img_width_out = CNN.__img_dim_after_convolution(conv1_img_width_out, max_pool1_kernel_size, 0,
                                                                   max_pool1_kernel_stride)
        max_pool1_img_height_out = CNN.__img_dim_after_convolution(conv1_img_height_out, max_pool1_kernel_size, 0,
                                                                    max_pool1_kernel_stride)

        conv2_img_width_out = CNN.__img_dim_after_convolution(max_pool1_img_width_out, conv2_kernel_size,
                                                               conv2_kernel_padding,
                                                               conv2_kernel_stride)
        conv2_img_height_out = CNN.__img_dim_after_convolution(max_pool1_img_height_out, conv2_kernel_size,
                                                                conv2_kernel_padding,
                                                                conv2_kernel_stride)
        max_pool2_img_width_out = CNN.__img_dim_after_convolution(conv2_img_width_out, max_pool2_kernel_size, 0,
                                                                   max_pool1_kernel_stride)
        max_pool2_img_height_out = CNN.__img_dim_after_convolution(conv2_img_height_out, max_pool2_kernel_size, 0,
                                                                    max_pool1_kernel_stride)

        # Fully connected layers
        fc1_in_features = conv2_out_channels * max_pool2_img_width_out * max_pool2_img_height_out
        self.fc1 = nn.Linear(fc1_in_features, fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)
        self.fc3 = nn.Linear(fc2_out_features, num_classes)

        self.max_pool1_kernel_size = max_pool1_kernel_size
        self.max_pool1_kernel_stride = max_pool1_kernel_stride

        self.max_pool2_kernel_size = max_pool2_kernel_size
        self.max_pool2_kernel_stride = max_pool2_kernel_stride

    @staticmethod
    def __img_dim_after_convolution(dim_size, conv_kernel_size, conv_kernel_padding, conv_kernel_stride):
        return int((dim_size - conv_kernel_size + 2 * conv_kernel_padding) / conv_kernel_stride + 1)

    def forward(self, x):
        # conv1 -> ReLU -> Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (self.max_pool1_kernel_size, self.max_pool1_kernel_stride))
        # conv2 -> ReLU -> Max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (self.max_pool2_kernel_size, self.max_pool2_kernel_stride))
        x = torch.flatten(x, 1)
        # Apply fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return x
