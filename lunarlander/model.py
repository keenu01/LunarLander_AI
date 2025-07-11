import torch as t
import torch.nn as nn
import torch.nn.functional as f


class Network(nn.Module):
    def __init__(self,input_layer=0,hidden_layer=0,output_layer=0):
        super(Network,self).__init__()
        self.net = nn.Sequential(nn.Linear(input_layer,hidden_layer),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layer,hidden_layer),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layer,output_layer),
                                )
    def forward(self,x):
            return self.net(x)
        
