import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN2(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN2, self).__init__()
        
        """
        formula [(W−K+2P)/S]+1.
        """
        self.conv1 = nn.Conv1d(input_size, 25, kernel_size=25, stride=1) #output: [(1000-25+2*0)/1]+1 = 976
        self.maxpool1 = nn.MaxPool1d(2, stride=2) #output: [((976+2*0-1*(2-1))-1)/2]+1= 488
        self.conv2 = nn.Conv1d(25,25,kernel_size=25, stride = 1) #output: [(488-25+2*0)/1]+1 = 464
        self.maxpool2 = nn.MaxPool1d(2, stride=2) #output: [((464+2*0-1*(2-1)-1)/2]+1= 232
        self.conv3 = nn.Conv1d(25,25,kernel_size=25, stride = 1) #output: [(232-25+2*0)/1]+1 = 208
        self.maxpool3 = nn.MaxPool1d(2, stride=2) #output: [(208+2*0-1*(2-1)-1)/2]+1= 104
        self.conv4 = nn.Conv1d(25,25,kernel_size=25, stride = 1) #output: [(104-25+2*0)/1]+1 = 80 
        self.maxpool4 = nn.MaxPool1d(2, stride=2) #output: [(80+2*0-1*(2-1)-1)/2]+1= 40
        self.leakyrelu=torch.nn.LeakyReLU()
        #self.fc1 = nn.Linear(25*83, output_size)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.leakyrelu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.leakyrelu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.leakyrelu(self.conv4(x))
        x = self.maxpool4(x)
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2])) #flatten
        output = x
        
        return output