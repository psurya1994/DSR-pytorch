# Loading required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

PHI_SIZE = 512

class thetaNet(nn.Module):
    """Encoder for abstracting the states.
    """

    def __init__(self, h=80, w=80, outputs=PHI_SIZE):
        """Constructor for encoder
        
        Inputs: 
            h -> height of input frame
            w -> width of input frame
            outputs -> size of abstracted states
        """
        super(thetaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Computing size of linear layer
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        """
        Inputs:
            x -> image of size b x 3 x h x w
            where b is the batch size
                  h, w are the height and width (80 by default)
                  
        Outputs:
            tensor of size b x 512
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)));
        return self.head(x.view(x.size(0), -1))
    
class theta2Net(nn.Module): 
    """Decoder for reconstructing the state.
    """
    def __init__(self, h=80, w=80, inputs=PHI_SIZE):
        """Constructor for the decoder
        
        Inputs: 
            h -> height of reconstructed frame
            w -> width of reconstructed frame
            inputs -> size of abstracted state space
        """
        super(theta2Net, self).__init__()
        
        # Computing size of linear layer
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        self.convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        self.convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = self.convw * self.convh * 64
        
        # Creating required layers for reconstruction
        self.linear = nn.Linear(inputs, linear_input_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, output_padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, output_padding=1)
    
    def forward(self, x):
        """
        Inputs:
            x -> tensor of size b x 512
            where b is the batch size
                  
        Outputs:
            tensor of size b x 3 x h x w
            (same size of the input so that MSE can be calculated)
        """
        x = F.relu(self.linear(x))
        x = x.view(x.size(0), 64, self.convh, self.convw)
        x = self.bn1(x)
        x = F.relu(self.bn2(self.deconv1(x)))
        x = F.relu(self.bn3(self.deconv2(x)))
        x = self.deconv3(x)
        return x
    
class alphaNet(nn.Module):
    """Network that finds successor features from abstracted states
    
    This network assumes three output actions.
    """

    def __init__(self, in_size=PHI_SIZE, mid_size=256, actions=3):
        """
        Inputs:
            in_size -> size of abstracted states (int)
            mid_size -> number of nodes in pre ultimate layer (int)
        """
        super(alphaNet, self).__init__()
        
        # Creating required layers for finding SRs
        self.head11 = nn.Linear(in_size, in_size)
        self.head12 = nn.Linear(in_size, mid_size)
        self.head13 = nn.Linear(mid_size, in_size)
        self.head21 = nn.Linear(in_size, in_size)
        self.head22 = nn.Linear(in_size, mid_size)
        self.head23 = nn.Linear(mid_size, in_size)
        self.head31 = nn.Linear(in_size, in_size)
        self.head32 = nn.Linear(in_size, mid_size)
        self.head33 = nn.Linear(mid_size, in_size)
        self.actions = actions

    def forward(self, x):
        """
        Inputs:
            x -> b x 512
        Returns:
            Sucessor features in the tensor b x Na x 512
            (where Na is the number of actions)
        """
        x1 = self.head13(F.relu(self.head12(F.relu(self.head11(x)))))
        x2 = self.head23(F.relu(self.head22(F.relu(self.head21(x)))))
        x3 = self.head33(F.relu(self.head32(F.relu(self.head31(x)))))
        y = torch.cat((x1, x2, x3), 1)
        return y.view(y.size(0), self.actions, -1)
    
class wNet(nn.Module):
    """Network that maps state abstraction to rewards
    """

    def __init__(self, in_size=PHI_SIZE):
        super(wNet, self).__init__()
        self.head = nn.Linear(in_size, 1)

    def forward(self, x):
        """
        Inputs:
            x -> b x 512 tensor
        Outputs:
            tensor of size b (CHECK: b x 1?)
        """
        return self.head(x)