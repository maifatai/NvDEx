import torch
from torch import nn
from torch.nn import functional as f

class ResBlk_3D(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk_3D, self).__init__()
        self.conv1=nn.Conv3d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm3d(ch_out)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm3d(ch_out)

        self.extra=nn.Sequential()
        if ch_out!=ch_in:
            self.extra=nn.Sequential(
                nn.Conv3d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm3d(ch_out)
                )
        self.act2=nn.ReLU(inplace=True)
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.act1(out)
        out=self.conv2(out)
        out=self.bn2(out)

        out=self.extra(x)+out
        out=self.act2(out)
        return out
class Attention(nn.Module):
    def __init__(self,channel,reduction=8):
        super(Attention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool3d(output_size=1)
        self.max_pool=nn.AdaptiveMaxPool3d(output_size=1)
        self.flatten=nn.Flatten()
        self.channel_attention=nn.Sequential(
            nn.Linear(in_features=channel,out_features=channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,out_features=channel,bias=False),
            nn.ReLU(inplace=True)
        )
        self.sig=nn.Sigmoid()

        self.spatial_attention=nn.Conv3d(in_channels=channel,out_channels=channel,kernel_size=3,stride=1,padding=1)#需要调整kernel

    def forward(self,x):
        b,c,_,_,_=x.size()

        cha_avg=self.avg_pool(x)
        cha_avg=self.flatten(cha_avg)
        cha_avg_attention=self.channel_attention(cha_avg).view(b,c,1,1,1)

        cha_max=self.max_pool(x)
        cha_max=self.flatten(cha_max)
        cha_max_attention=self.channel_attention(cha_max).view(b,c,1,1,1)

        cha_add=cha_avg_attention+cha_max_attention
        cha_add=self.sig(cha_add)

        cha_attention=torch.mul(x,cha_add)

        spa_attention=self.spatial_attention(cha_attention)
        spa_attention=self.sig(spa_attention)
        attention=torch.mul(cha_attention,spa_attention)

        return attention



class ResNet34(nn.Module):
    def __init__(self,class_num):
        super(ResNet34, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),#padding=3是因为卷积为7，当卷积的尺寸为1
            nn.BatchNorm3d(num_features=64)
        )
        self.attention1=Attention(64)
        self.pool1=nn.MaxPool3d(kernel_size=3,stride=2,padding=1)


        self.blk1=ResBlk_3D(64,64)
        self.blk2=ResBlk_3D(64,64)
        self.blk3=ResBlk_3D(64,64)

        self.blk4=ResBlk_3D(64,128,2)
        self.attention2=Attention(128)
        self.blk5=ResBlk_3D(128,128)
        self.blk6 = ResBlk_3D(128, 128)
        self.blk7 = ResBlk_3D(128, 128)

        self.blk8=ResBlk_3D(128,256,2)
        self.attention3=Attention(256,16)
        self.blk9=ResBlk_3D(256,256)
        self.blk10 = ResBlk_3D(256, 256)
        self.blk11= ResBlk_3D(256, 256)
        self.blk12= ResBlk_3D(256, 256)
        self.blk13= ResBlk_3D(256, 256)

        self.blk14=ResBlk_3D(256,512,2)
        self.attention4=Attention(512,32)
        self.blk15=ResBlk_3D(512,512)
        self.blk16=ResBlk_3D(512,512)

        self.glob_avg_pool=nn.AdaptiveAvgPool3d(1)

        self.flatten=nn.Flatten()
        self.lin1=nn.Linear(in_features=512,out_features=64)
        self.relu=nn.ReLU(inplace=True)
        self.lin2=nn.Linear(in_features=64,out_features=2)
        self.sig=nn.Sigmoid()
    def forward(self,x):
        b,c,_,_,_=x.size()
        out=f.relu(self.conv1(x))
        out=self.attention1(out)
        out=self.pool1(out)
        out=self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)

        out = self.blk4(out)
        out = self.attention2(out)
        out = self.blk5(out)
        out = self.blk6(out)
        out = self.blk7(out)

        out=self.blk8(out)
        out = self.attention3(out)
        out = self.blk9(out)
        out = self.blk10(out)
        out = self.blk11(out)
        out = self.blk12(out)
        out = self.blk13(out)

        out = self.blk14(out)
        out = self.attention4(out)
        out = self.blk15(out)
        out = self.blk16(out)

        out=self.glob_avg_pool(out)
        out=self.flatten(out)
        out=self.lin1(out)
        out=self.relu(out)
        out=self.lin2(out)
        out=self.sig(out)
        return out

if __name__=="__main__":
    x=torch.randn(4,3,5,6,7)#第一个是batch，第二个是channel，之后是长宽高
    net=ResBlk_3D(3,10)
    out=net(x)
    print(out.shape)

    x1=torch.randn(4,3,224,224,224)
    net1=ResNet34(2)
    out1=net1(x1)
    print(out1.shape)
    p = sum(map(lambda p:p.numel(), net1.parameters()))#输出网络的总参数
    print('parameters size:', p)

    # net2=Attention(3,1)
    # out2=net2(x1)
    # print(out2.shape)
