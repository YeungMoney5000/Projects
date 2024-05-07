import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torchvision import transforms
import numpy as np
import random
global device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self,n_input, n_output,padding = 1, stride =1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d( n_input, n_output, kernel_size = 3, padding = padding,stride = stride, bias=False),
                nn.BatchNorm2d(n_output),
                nn.Dropout(p=.1),
                torch.nn.ReLU(),
                torch.nn.Conv2d( n_output, n_output, kernel_size = 3, padding = padding,stride = 1, bias=False),
                nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
            )

            nn.init.xavier_uniform_(self.net[0].weight)
            nn.init.xavier_uniform_(self.net[4].weight)

            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))
        def forward(self, x):
            sample = x
            if self.downsample != None:
                sample = self.downsample(sample)

            return self.net(x) + sample

    class Decoder(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.conv_trans = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                         stride=stride, output_padding=1)

        def forward(self, x, prev_features):
            z= self.conv_trans(x)
            z = z[:, :, :prev_features.size(2), :prev_features.size(3)]
            z = torch.cat([z, prev_features], dim=1)
            return z
        


    def __init__(self,layers = [64,128], n_input_channels=3, n_output_channels = 1, kernel_size=3):
        super().__init__()
        self.orig_layer= [32,64,128]
        self.prev_layer = [3,32,64]
        self.input_mean = torch.Tensor([0.3234, 0.3310, 0.3444])
        self.input_std = torch.Tensor([0.2524, 0.2219, 0.2470])
        L= [
            torch.nn.Conv2d(n_input_channels, 32, kernel_size = 3, padding = 1,stride = 2, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(p=.1),
            torch.nn.ReLU(),
        ]
        c = 32
        for l in layers:
            L.append(self.Block(c,l,stride=2))
            c=l
        for i, l in list(enumerate(self.orig_layer))[::-1]:
            L.append(self.Decoder(c, l,stride = 2))
            c = l + self.prev_layer[i]
        self.network = torch.nn.Sequential(*L)
        self.heatmap = torch.nn.Conv2d(c, n_output_channels, 1)
        #self.sizemap = torch.nn.Conv2d(c, 2, 1)
        nn.init.xavier_uniform_(self.heatmap.weight)
        #nn.init.xavier_uniform_(self.sizemap.weight)
        #raise NotImplementedError('CNNClassifier.__init__')
    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64)) 
        @return: torch.Tensor((B,5))
        """
        i=0
        prev_features = []
        x=x.to(device)
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device).to(x.device)
        self.network=self.network.to(device)
        self.heatmap = self.heatmap.to(device)
        #self.sizemap = self.sizemap.to(device)
        for module in self.network:
            if isinstance(module, self.Decoder):
                z = module(z, prev_features[i-1])
                i-=1
            elif isinstance(module, self.Block) or isinstance(module, torch.nn.Conv2d):
                prev_features.append(z)
                z = module(z)
                i+=1
            else:
                z = module(z)
        z = self.heatmap(z)
        z = torch.squeeze(z,1)
        return spatial_argmax(z)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
