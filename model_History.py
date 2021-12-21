import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class VIN(nn.Module):
    def __init__(self, config):
        super(VIN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.h1 = nn.Conv2d(
            in_channels=config.l_i, #4
            out_channels=config.l_h1, #50
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        # self.h2 = nn.Conv2d(
        #     in_channels=config.l_h1,
        #     out_channels=config.l_h2, #50
        #     kernel_size=(5, 5),
        #     stride=1,
        #     padding=2,
        #     bias=True)     
        # self.h3 = nn.Conv2d(
        #     in_channels=config.l_h3,
        #     out_channels=config.l_h3, #50
        #     kernel_size=(5, 5),
        #     stride=1,
        #     padding=2,
        #     bias=True)   
        self.r = nn.Conv2d(
            in_channels=config.l_h1,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)

        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=config.l_q, #10
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.fc = nn.Linear(in_features=config.l_q, out_features=8, bias=False)
        self.w = Parameter(
            torch.zeros(config.l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, X_input, S1, S2, config):

        # X_input of size(batch, 4, 16, 16)
        h1 = self.h1(X_input) # hiden layer, encode map&target information
        # h2 = self.h2(h1)
        # h3 = self.h3(h2)
        r = self.r(h1) # f_R reward

        q = self.q(r) # of size(batch, f_A(A_bar), domsize[0](S_x), domsize[1](S_y)) = Q(S, A_bar)

        v, _ = torch.max(q, dim=1, keepdim=True) # of size(batch, 1, domsize[0], domsize[1])
        for i in range(0, config.k - 1):
            # q of size(128,10,16,16), Q(S, A_bar) where S has 16*16 states, A_bar has 10 actions
            q = F.conv2d(
                torch.cat([r, v], 1),                     # of size(128,2,16,16)
                torch.cat([self.q.weight, self.w], 1),    # of size(10,2,3,3)  [r,v] conv ([weight,w].T)
                stride=1,
                padding=1)
            v, _ = torch.max(q, dim=1, keepdim=True)

        # get Q_final(S, A_bar)
        q = F.conv2d(                               
            torch.cat([r, v], 1),                   
            torch.cat([self.q.weight, self.w], 1),  
            stride=1,
            padding=1)

        # get Q_final(S=s_xy, A_bar)
        slice_s1 = S1.long().expand(config.imsize, 1, config.l_q, q.size(0)) # (128,) -> (16,1,10,128)
        slice_s1 = slice_s1.permute(3, 2, 1, 0) # of size(128,10,1,16)
        q_out = q.gather(2, slice_s1).squeeze(2) # (128,10,16,16) -> (128,10,1,16) -> (128,10,16)

        slice_s2 = S2.long().expand(1, config.l_q, q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0) # of size(128,10,1)
        q_out = q_out.gather(2, slice_s2).squeeze(2) # (128,10,16) -> (128,10,1) -> (128,10) 

        # get Q_pred(S=s_xy, A_pred)
        logits = self.fc(q_out) # of size(128,8)

        return logits, self.sm(logits), v
