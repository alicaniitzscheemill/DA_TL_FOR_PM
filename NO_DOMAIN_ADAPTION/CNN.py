import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN model
    """
    
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        
        """
        formula [(W−K+2P)/S]+1.
        """
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=100, stride=1)#input: 1024
        self.conv2 = nn.Conv1d(64,32,kernel_size=10, stride = 1, padding=1)#input: [(1024-100+2*0)/1]+1 = 925
        self.batch1 =nn.BatchNorm1d(32)#input: [(925-10+2*1)/1]+1 = 918
        self.conv3 = nn.Conv1d(32,32,kernel_size=5, stride = 1, padding=1) #input:918
        self.batch2 =nn.BatchNorm1d(32)#input: [(918-5+2*1)/1]+1 = 916
        self.fc1 = nn.Linear(32*916, output_size)

    def forward(self, x):
        x = F.selu(self.conv1(x)) #conv1
        x = self.conv2(x) #conv2
        x = F.selu(self.batch1(x)) #batch1
        x = self.conv3(x) #conv3
        x = F.selu(self.batch2(x)) #batch2
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2])) #flatten
        x = self.fc1(x) #linear1
        output = x
        
        return output