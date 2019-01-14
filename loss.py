# reference: https://arxiv.org/abs/1812.01187
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSCritierion(nn.Module):
    """
    label smoothing loss with Cross Entropy
    """
    def __init__(self, eta):

        super().__init__()
        self.eta = eta
        assert self.eta > 0, 'eta should be larger than 0'


    def forward(self, input, target):
        batchsize, num_class = input.size()

        one_hot_target = torch.ones_like(input) * self.eta / (num_class-1)
        one_hot_target.scatter_(1,target.unsqueeze(1),1-self.eta)

        log_softmax = F.log_softmax(input, dim=1)
        loss = - 1 / batchsize * (log_softmax.mul(one_hot_target)).sum()

        return loss



if __name__ == '__main__':
    input = torch.rand(3,5)
    target = torch.tensor([1, 2, 3])
    print('input:', input)

    #Label Smoothing loss
    loss_func = LSCritierion(0.0001)
    input1 = Variable(input,requires_grad=True)
    loss = loss_func(input1,target)
    print('label smoothing loss:',loss)
    loss.backward()
    print('grad:',input1.grad)

    #CrossEntropyLoss
    input2 = Variable(input,requires_grad=True)
    loss_func2 = nn.CrossEntropyLoss()
    loss2 = loss_func2(input2,target)
    print('cross entropy loss:',loss2)
    loss2.backward()
    print('grad:', input2.grad)


