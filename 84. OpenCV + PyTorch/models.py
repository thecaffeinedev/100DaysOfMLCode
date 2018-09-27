## TODO: define the convolutional neural network architecture
# NOTE: See below commented out section for in depth explanation of how to calculate input and output shapes

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # not adding padding - therefore need to ensure images are divisible by 5 to avoid columns being lost (as pyTorch leaves 
        # it up to user to pad)
        # see: http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        # made the kernel size 4x4 as 224 is divisible by 4. But note that this will decrease the dimension by 3. Therefore after 
        # the first convolutional layer I'll have a 220x220 image. See analysis in below triple quote commented out section. Basically,
        # we keep striding by 1 (as that is default) and when I cannot fit a full 4x4 box it will stop. And the # of full 4x4 boxes that can 
        # be fit in using a stride length of 1 is the output dimension of the convolutional layer
        # Also, note that I'm slowly increasing the number of filters in each convolutional layer to hopefully get better performance as 
        # at first start only filter for a handful of distinguishing features and then hopefully build on those more basic 'objects' to get 
        # more and more distinguishing features until we reach to 32
        self.conv1 = nn.Conv2d(1, 8, 4)
        self.conv2 = nn.Conv2d(8, 16, 4)
        self.conv3 = nn.Conv2d(16, 32, 4)
        # kernel size of (2,2) and since left out stride it defaults to the size of the kernel. Note: putting a pooling layer after
        # every convolutional layer just like we did in Keras CV capstone. Also, note that since every pooling layer will have a pool size / kernel size
        # of 2x2. We don't have to make another MaxPool2d object for each of them. Can just reuse.
        self.maxPool = nn.MaxPool2d(2)
        # Now in the forward() method we will flatten the input to a 'flat' vector per image within the batch. This means if batch_size is 10 then
        # we will flatten each image within the 10 in the batch to a flat vector. But the batch itself will still be a 2D matrix with 
        # dimension (10, whatever each flattened input dimension is)
        # Therefore, in our example this will be:
        # (224, 224, 1) - > (221, 221, 8) -> (110, 110, 8) -> (107, 107, 16) -> (53, 53, 16) -> (50, 50, 32) -> (25, 25, 32) -> 25*25*32 = 20,000
        # Above structure assumes Input->conv1->maxPool->conv2->maxPool->conv3->maxPool->Flatten->Linear
        self.linear1 = nn.Linear(20000, 5000)
        self.linear2 = nn.Linear(5000, 2500)
        self.linear3 = nn.Linear(2500, 500)
        # some more NOTEs on Linear layers:
        # note: http://pytorch.org/docs/master/nn.html#linear.....the docs say the in_features is the value of the size of EACH input sample. This
        # is a key thing to not overlook. This is the size of the input layer when it's feeding forward and it only feeds one image at a time.
        # therefore this value is the dimension of each individual input sample and in our case x.size())[1] because x.size())[0] = batch size 
        # and x.size())[1] is the dimension of each individual image within the batch 
        # then NOTE: that the out_features parameter is also the size of EACH input sample output that we want. I.e what we want each layer to 
        # output for each individual layer. Therefore, after this first linear layer its output is 10x5000 


        # now we output 136 nodes as that's the output size we want. As we have 68 keypoints and each keypoint has a 'x' and 'y' component
        self.output_layer = nn.Linear(500, 136)
        # do dropout between all the linear layers
        self.drop1 = nn.Dropout2d(p=.3)
        self.drop2 = nn.Dropout2d(p=.15)
        self.drop3 = nn.Dropout2d(p=.10)


        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and layers to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # note: X starts of as the input image. And then just like we do in keras we keep using it as the output of each 
        # subsequent layer
        # get the number of examples within the input batch given X. (X is iterable of inputs. ie a variable IMAGES that has a bunch of 
        # images within it to be used in the network)
        batch_size = x.size()[0]
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = self.maxPool(F.relu(self.conv3(x)))
        # this should be like a Flatten layer. Where putting just the argument '-1' just takes the input layer dimensions and then 
        # just flattens it accordingly. (it works like numpy's 'reshape' method. batch_size is the first argument as we want it 
            # to have 10 'columns' / rows and then flat within each of those. We want that as to keep each individual input image
            # seperate)
        x = x.view(batch_size, -1)
        # print(x.size())
        # note: this is how we make the 'Dense' (fully connected) layers like we did in Keras. There are called Linear here because remember
        # dense layer is a layer that is a linear operation of all the input vectors. We then will add on in the 'forward' function the 
        # activation function we want to use on this layer. Remember the last layer with the output of 136 values per input sample 
        # (note: per input sample means that we'll be outputting a 2-D output 10x136. Where 10 is just the batch_size - happens to be 10 in this
        # project, but if you change it in the juptyer notebook2 then it will change. And this 10 represents 10 images in each batch put into
        # this network. Therefore, the 136 output layer is the 136 'keypoints' one for x and one for y coordinate of each of the 68 keypoints.) 
        # will have no activation function because we aren't classifying but really more doing a regression. While the other ones we are classifying features within 
        # the face to help us determine facial keypoints


        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        x = F.relu(self.linear2(x))
        x = self.drop2(x)
        x = F.relu(self.linear3(x))
        x = self.drop3(x)
        x = self.output_layer(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x




"""Below is a models.py from another submission (that worked). Used it to see how the input shape / output shape was calculated
NOTE: he gets that the first linear layer has an input size of 16,928 because 
(1) We for the Linear layer we have to flatten out input from the previous layer that is a 3-D input of the dimension of the 
resulting image after pooling convolving and the 'output' channels (second argument of Conv2d)
(2) Network structure below is Conv1 -> pool -> Conv2 -> pool -> Conv3 -> pool -> Linear
Therefore, the input shape to Conv1 is (224, 224, 1). And then after Conv1 with a kernel size of (3, 3) (3rd argument of Conv2d) 
and an output channel of 32 (i.e make 32 different 'filters' for the network to identify to help detect objects) we get a shape
of (222, 222, 32). We get (222, 222) because padding is defaulted to 0 and thus when we are going over the image and the kernel
'hangs' over the edge of the current image then we don't add padding to it, rather we just leave that kernel out of our result convolved image
ex.) if we have a (5, 5) image and our kernel size is (2,2) then the resulting output of that convolutional layer will be 
(4, 4) - assuming no padding and a stride length of 1. Try making a 4x4 grid. Then take a 2x2 box and move it 1 square at a time. You'll only
be able to get 4 full 2x2 boxes in the height and width direction which leaves us with a (4,4) output. Therefore, I think the formula for the 
output shape of a convolutional layer given the input is just width = input_width - (kernel_width - 1), height = input_height - (kernel_height - 1)
(3) And since each pooling layer has the same 'pool_size' ie pooling kernel_size of a 2x2. Then each pooling layer halves the input size (take 
    the floor again if get a decimal).
(4) Therefore,  

Format of below -> First the layer is listed and then the shape of the 'image' after going through that layer.
Input (224, 224, 1) -> Conv1 (222, 222, 6) -> pool (111, 111, 32) -> Conv2 (107, 107, 16) -> pool (53, 53, 16) -> Conv3 (46, 46, 32) -> pool (23, 23, 32)
-> (23*23*32 = 16,928) Linear -> (512) -> note this output shape from the first Linear layer is totally controlled by user

## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn

import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 8)
        self.fc1 = nn.Linear(16928, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 136)
        self.drop_out = nn.Dropout2d(0.3)
        self.batchn1 = nn.BatchNorm2d(6)
        self.batchn2 = nn.BatchNorm2d(16)
        self.batchn3 = nn.BatchNorm2d(32)
        #self.fc3 = nn.Linear(84, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and layers to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.batchn1(self.conv1(x))))
        #x = self.drop_out(x)
        x = self.pool(F.relu(self.batchn2(self.conv2(x))))
       
        #x = x.view(x.size()[0], self.num_flat_features(x))
        x = self.pool(F.relu(self.batchn3(self.conv3(x))))
        x = self.drop_out(x)
        batch_size = x.size()[0]
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x=F.relu(self.conv1(x)))
        #print("after all {}".format(x.size()))
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

"""