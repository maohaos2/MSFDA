import numpy as np
import torch
import torch.nn as nn

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def mutual_info_loss(prob1):
    entropy = torch.mean(Entropy(prob1))
    prob1_mean = torch.mean(prob1,0)
    div = torch.sum(- prob1_mean * torch.log(prob1_mean) )
    loss = entropy-div
    return loss

def prototype_weights(feature, center, bandwidth):
    output = torch.zeros(feature.size()[0], center.size()[0])
    for i in range(center.size()[0]):
        output[:,i] = -torch.sum((feature-center[i])**2, -1)
    softmax_ = nn.Softmax(dim=1)(output/(bandwidth*2))
    return softmax_