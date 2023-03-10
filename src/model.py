import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, filter_num=5, padding=True,):
        super(UNet, self).__init__()

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (filter_num + i), padding)
            )
            prev_channels = 2 ** (filter_num + i)
        self.midconv = nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (filter_num + i), padding)
            )
            prev_channels = 2 ** (filter_num + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.midconv(x), negative_slope = 0.1)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetUpBlock, self).__init__()

        self.up = nn.Sequential(
                nn.Upsample(mode='bicubic', scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            )
        self.conv_block = UNetConvBlock(in_size, out_size, padding)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
    
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]
    
    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat((up, crop1), 1)
        out = self.conv_block(out)
    
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.first_flow = UNet(6,4,5)
        self.refine_flow = UNet(10,4,4)
        self.weight_map = UNet(16,2,4)
        self.final = UNet(9,3,4)

    def warp(self, img, flow):
        _, _, H, W = img.size()
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        gridX = torch.tensor(gridX, requires_grad=False).to(device)
        gridY = torch.tensor(gridY, requires_grad=False).to(device)
        u = flow[:,0,:,:]
        v = flow[:,1,:,:]
        x = gridX.unsqueeze(0).expand_as(u).float() + u
        y = gridY.unsqueeze(0).expand_as(v).float() + v
        normx = 2*(x / W - 0.5)
        normy = 2*(y / H - 0.5)
        grid = torch.stack((normx, normy), dim=3)
        warped = F.grid_sample(img, grid, align_corners=True)

        return warped

    def process(self, frame0, frame1, t):
        x = torch.cat((frame0, frame1), 1)
        flow = self.first_flow(x)
        flow_0_1, flow_1_0 = flow[:,:2,:,:], flow[:,2:4,:,:]
        flow_t_0 = -(1-t) * t * flow_0_1 + t * t * flow_1_0
        flow_t_1 = (1-t) * (1-t) * flow_0_1 - t * (1-t) * flow_1_0

        flow_t = torch.cat((flow_t_0, flow_t_1, x), 1)
        flow_t = self.refine_flow(flow_t)
        flow_t_0 = flow_t_0 + flow_t[:,:2,:,:]
        flow_t_1 = flow_t_1 + flow_t[:,2:4,:,:]

        xt1 = self.warp(frame0, flow_t_0)
        xt2 = self.warp(frame1, flow_t_1)

        temp = torch.cat((flow_t_0, flow_t_1, x, xt1, xt2), 1)
        mask = torch.sigmoid(self.weight_map(temp))
        w1, w2 = (1-t) * mask[:,0:1,:,:], t * mask[:,1:2,:,:]
        
        output = (w1 * xt1 + w2 * xt2) / (w1 + w2 + 1e-8)

        return output, flow_t_0, flow_t_1, w1, w2
    
    def forward(self, frame0, frame1, t=0.5):
        output, flow_t_0, flow_t_1, w1, w2 = self.process(frame0, frame1, t)
        compose = torch.cat((frame0, frame1, output), 1)
        final = self.final(compose) + output
        final = final.clamp(0,1)
        
        return final, flow_t_0, flow_t_1, w1, w2


def normal_init(m, mean, std):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        c = 64
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=c*4, out_channels=c*8, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=c*8, out_channels=1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    def forward(self, first, mid, last):
        x = torch.cat([first, mid, last], dim=1)
        x = self.model(x)

        return x 
