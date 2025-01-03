import torch
import torch.nn as nn
from torchsummary import summary

class Bottleneck(nn.Module): # Bottleneck module as a single class which will be used to create the ResNet model. Each bottleneck as 3 convolutions. 
    expansion = 4 # sets how much the bottleneck will expand the output channels of the last layer in a bottleneck block to. Used 4 as per the original paper. 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # note this is the convolution which will use a stride of 2 to reduce the image size. This happens in the first block of layers 2, 3 and 4 only. All other convolutions in all blocks in each of the layers use a stride of 1, as per the ResNet model. 
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False) # this is the convolution where number of channels is expanded, as per the ResNet model. 
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True) # this will modify the original tensor rather than operating on a copy. Significant memory savings as this module is the fundamental repeating unit. 
        self.downsample = downsample # helps match the input dimensions to the dimensions after convolution for the special skip connection.

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) #not applied ReLU() here because we don't want to remove the negatives just yet, as the next step is addition with the original image. In order to learn the residual F(x) correctly, negatives will be needed in the tensor now. The residual function F(x) should learn both positives and negatives now. 

        # Special skip connection - triggered only in the first block of all layers, where we need to downsample the dimensions and channels of input x to match those of F(x) after convolutions, to be able to add them up.
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # The core ResNet addition is here; H(x) = F(x) + x. Skip connection by virtue of adding identity variable, which is the original input without convolutions.If special skip connection, downsampling will be applied.
        out = self.relu(out) # ReLU() finally applied here, after the addition. Now that the original input x and residual F(x) have been added, we can safely remove the negatives.  

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000): # num_classes to be set as per the dataset. 10 for CIFAR-10, 1000 for ImageNet 1k.
        super(ResNet50, self).__init__()
        self.in_channels = 64 # only used for the initiation of the first bottleneck block in the first layer. 
        
        ## See Excel sheet for Model Architecture
        
        # Adjusted Initial Conv Layer for ImageNet 1k
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #kernel size is 7 here for ImageNet 1k.
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # as before, this will modify the input tensor. Good memory savings here as the input image will be large in size here. 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Add max pooling layer

        # Layers with Bottleneck Blocks 
        self.layer1 = self._make_layer(Bottleneck, 64, 3) # stride is 1 here, so the downsampling will only adjust for the channel size in the first block of this layer
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2) #stride input here used only in the first block of the layer for downsampling. This layer onwards, downsampling will adjust for both the image size and channel dimensions. 
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Final Layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion: # triggered for all layers - for layer 1 this only adjusts the channel size as stride is 1, for layers 2,3 and 4 this adjusts both the channel size and dimensions with the stride. 
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # first bottleneck block of the layer which takes the stride and downsample inputs
        self.in_channels = out_channels * block.expansion # sets the number of input channels for every subsequent block as the expanded output of the bottleneck from the first block. This is squeezed again by the first convolution of the new block, and expanded again by the last layer, continuing the trend.
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels)) # all subsequent bottleneck blocks take default stride and downsample inputs, so no downsampling happens in any of the later blocks of any layer.

        return nn.Sequential(*layers) #unpack the list layers[] defined above so that each block in the list is passed as a separate argument to nn.sequential, effectively creating all the blocks for a layer this function is called for. 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Add max pooling layer in forward pass

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNet50().to(device)
    summary(model, input_size=(3, 224, 224)) # size is (3, 224, 224) for ImageNet 1k.
