import traceback
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from katago.train import load_model

class ResBlock(nn.Module):
    def __init__(self,num_channels,scale_init):
        super(ResBlock, self).__init__()
        kernel_size = 3
        self.biasa = nn.Parameter(torch.zeros(num_channels,1,1))
        self.conva = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=1, bias=False)
        torch.nn.init.normal_(self.conva.weight,std=math.sqrt(2.0 / num_channels / kernel_size / kernel_size)*scale_init)
        self.biasb = nn.Parameter(torch.zeros(num_channels,1,1))
        self.scalb = nn.Parameter(torch.ones(num_channels,1,1))
        self.convb = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=1, bias=False)
        torch.nn.init.zeros_(self.convb.weight)

    def forward(self, trunk):
        x = F.relu(trunk+self.biasa)
        x = self.conva(x)
        x = F.relu(x*self.scalb+self.biasb)
        x = self.convb(x)
        return trunk+x

class GPoolResBlock(nn.Module):
    def __init__(self,num_channels,scale_init):
        super(GPoolResBlock, self).__init__()
        kernel_size = 3
        self.biasa = nn.Parameter(torch.zeros(num_channels,1,1))
        self.conva = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=1, bias=False)
        torch.nn.init.normal_(self.conva.weight,std=math.sqrt(1.0 / num_channels / kernel_size / kernel_size)*scale_init)
        self.convg = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=1, bias=False)
        torch.nn.init.normal_(self.convg.weight,std=math.sqrt(1.0 / num_channels / kernel_size / kernel_size)*math.sqrt(scale_init))
        self.matg = nn.Parameter(torch.zeros(num_channels,num_channels))
        torch.nn.init.normal_(self.matg,std=math.sqrt(1.0 / num_channels)*math.sqrt(scale_init))
        self.biasb = nn.Parameter(torch.zeros(num_channels,1,1))
        self.scalb = nn.Parameter(torch.ones(num_channels,1,1))
        self.convb = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=1, bias=False)
        torch.nn.init.zeros_(self.convb.weight)

    def forward(self, trunk):
        x = F.relu(trunk+self.biasa)
        x = self.conva(x)
        g = self.convg(x)
        gsize = g.size()
        g = torch.sum(g,(2,3)) / (gsize[2] * gsize[3]) # nchw -> nc
        g = torch.matmul(g,self.matg)
        g = g.view(gsize[0],gsize[1],1,1)
        x = x + g
        x = F.relu(x*self.scalb+self.biasb)
        x = self.convb(x)
        return trunk+x


class Model(nn.Module):
    def __init__(self, num_channels, num_blocks):
        super(Model, self).__init__()
        # Channel 0: Next inference point
        # Channel 1: On-board
        # Channel 2: Black
        # Channel 3: White
        # Channel 4: Unknown
        # Channel 5: Turn number / 100
        # Channel 6: Noise stdev in turn number / 50
        # Channel 7: Source

        self.inference_channel = 0
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.conv0 = nn.Conv2d(in_channels=8, out_channels=self.num_channels, kernel_size=3, padding=1, bias=False)

        self.blocks = nn.ModuleList([])
        self.fixup_scale_init = 1.0 / math.sqrt(self.num_blocks)
        self.blocks.append(ResBlock(self.num_channels,self.fixup_scale_init))
        self.blocks.append(ResBlock(self.num_channels,self.fixup_scale_init))

        next_is_gpool = True
        for b in range(num_blocks-2):
            if next_is_gpool:
                self.blocks.append(GPoolResBlock(self.num_channels,self.fixup_scale_init))
            else:
                self.blocks.append(ResBlock(self.num_channels,self.fixup_scale_init))
            next_is_gpool = not next_is_gpool

        assert(len(self.blocks) == self.num_blocks)

        self.endtrunk_bias_focus = nn.Parameter(torch.zeros(self.num_channels,1,1))
        self.endtrunk_bias_g = nn.Parameter(torch.zeros(self.num_channels,1,1))
        self.convg = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=1, padding=0, bias=False)

        self.fc1 = nn.Linear(self.num_channels*2, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels,3)
        self.convaux = nn.Conv2d(in_channels=self.num_channels, out_channels=3, kernel_size=1, padding=0, bias=True)

    def forward(self, inputs):
        trunk = self.conv0(inputs)
        for i in range(self.num_blocks):
            trunk = self.blocks[i](trunk)

        head_focus = F.relu(trunk+self.endtrunk_bias_focus)
        head_g = F.relu(trunk+self.endtrunk_bias_g)
        aux = self.convaux(head_focus)
        gsize = head_g.size()

        x = torch.sum(head_focus * inputs[:,self.inference_channel:self.inference_channel+1,:,:],(2,3))
        g = torch.sum(head_g,(2,3)) / (gsize[2] * gsize[3]) # nchw -> nc

        x = torch.cat((x,g),dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x,aux

    def save_to_file(self, filename):
        state_dict = self.state_dict()
        data = {}
        data["num_channels"] = self.num_channels
        data["num_blocks"] = self.num_blocks
        data["state_dict"] = state_dict
        torch.save(data, filename)

    @staticmethod
    def load_from_file(filename):
        data = torch.load(filename)
        model = Model(data["num_channels"], data["num_blocks"])
        model.load_state_dict(data["state_dict"])
        return model
