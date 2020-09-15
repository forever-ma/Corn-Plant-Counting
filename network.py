import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import collections


class CORNNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CORNNet,self).__init__()
        self.frontend_feat=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat=[512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 1152,dilation = True)
        self.output_layer1 = CAMModule(512)
        self.output_layer2 = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv1_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv3_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv3_2=nn.Conv2d(512,512,kernel_size=1,bias=False)

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()           
            fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key=list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key]=list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)
    def forward(self,x):
        fv = self.frontend(x)
        #S=1
        ave1=nn.functional.adaptive_avg_pool2d(fv,(1,1))
        ave1=self.conv1_1(ave1)
        s1=nn.functional.upsample(ave1,size=(fv.shape[2],fv.shape[3]),mode='bilinear')
        c1=s1-fv
        w1=self.conv1_2(c1)
        w1=nn.functional.sigmoid(w1)

        #S=3
        ave3=nn.functional.adaptive_avg_pool2d(fv,(3,3))
        ave3=self.conv3_1(ave3)
        s3=nn.functional.upsample(ave3,size=(fv.shape[2],fv.shape[3]),mode='bilinear')
        c3=s3-fv
        w3=self.conv3_2(c3)
        w3=nn.functional.sigmoid(w3)

        fi=(w1*s1+w3*s3)/(w1+w3+0.000000000001)
        x1=torch.cat((fv,fi),1)
        x2 = self.output_layer1(fv)
        x3 = torch.cat((x1,x2), 1)
        x4 = self.backend(x3)
        y = self.output_layer2(x4)
        return y
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class CAMModule(nn.Module):
    def __init__(self, inn):
        super(CAMModule, self).__init__()
        base = inn // 4
        self.conv_ca = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bias=False),
                                     CAM(base),
                                     Conv2d(base, base, 3, same_padding=True, bias=False)
                                     )

    def forward(self, x):
        ca_feat = self.conv_ca(x)
        return ca_feat

class CAM(nn.Module):
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.para_mu = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        N, C, H, W = x.size()
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = F.softmax(energy,dim=-1)
        proj_value = x.view(N, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)

        out = self.para_mu*out + x
        return out

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


