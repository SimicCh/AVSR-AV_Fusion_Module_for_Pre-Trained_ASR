import torch
import torch.nn as nn



class ResidualBlock_specFront_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class spec_frontend_CNN(nn.Module):
    def __init__(self, processing_channels, layerNum, inp_dim, out_dim):
        super().__init__()
        self.processing_channels = processing_channels
        self.layerNum = layerNum
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.CNN_layer = nn.ModuleList( [ResidualBlock_specFront_CNN(in_channels=1, out_channels=self.processing_channels, downsample=nn.Conv2d(1, self.processing_channels, 1))] +
                                        [ResidualBlock_specFront_CNN(in_channels=self.processing_channels, out_channels=self.processing_channels) for _ in range(self.layerNum-2) ] +
                                        [ResidualBlock_specFront_CNN(in_channels=self.processing_channels, out_channels=1, downsample=nn.Conv2d(self.processing_channels, 1, 1))])
        self.out_proj = nn.Linear(self.inp_dim,self.out_dim)
    
    def forward(self, inp_specs):
        x = inp_specs[:,None,:,:]
        for layer in self.CNN_layer:
            x = layer(x)
        x = x.squeeze(1)
        x = self.out_proj(x)
        return x




class ResidualBlock_2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock_2D, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1)),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
                        nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResidualBlock_3D(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, downsample=None):
        super(ResidualBlock_3D, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, inter_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                        nn.BatchNorm3d(inter_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_channels, inter_channels, kernel_size=(5,3,3), stride=(1,1,1), padding=(2,1,1)),
                        nn.BatchNorm3d(inter_channels),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv3d(inter_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                        nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class LIPNET_CNN(nn.Module):
    def __init__(self, layers: list(), emb_size=64):
        super(LIPNET_CNN, self).__init__()

        self.layers = layers
        self.emb_size = emb_size
        self.inplanes = 64
        self.resblock2D = ResidualBlock_2D
        self.resblock3D = ResidualBlock_3D


        self.inpConv = nn.Sequential(
                        nn.Conv3d(1, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3)),
                        nn.BatchNorm3d(64),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)))
        
        self.layer01 = self._make_layer_res2D(self.resblock2D, 64, self.layers[0][0], stride=1)
        self.layer02 = self._make_layer_res3D(self.resblock3D, 64, 64, self.layers[0][1])
        
        self.layer11 = self._make_layer_res2D(self.resblock2D, 128, self.layers[1][0], stride=2)
        self.layer12 = self._make_layer_res3D(self.resblock3D, 128, 64, self.layers[1][1])

        # self.upsample = nn.Upsample(scale_factor=(self.sequlen_factor, 1, 1), mode='nearest')
        # Only capable for sequlen_factor=2
        self.upsample1 = nn.ConvTranspose3d(128, 128, (6, 1, 1), stride=(2, 1, 1), padding=(2, 0, 0))
        
        self.layer21 = self._make_layer_res2D(self.resblock2D, 256, self.layers[2][0], stride=2)
        self.layer22 = self._make_layer_res3D(self.resblock3D, 256, 64, self.layers[2][1])

        self.upsample2 = nn.ConvTranspose3d(256, 256, (6, 1, 1), stride=(2, 1, 1), padding=(2, 0, 0))
        
        self.layer31 = self._make_layer_res2D(self.resblock2D, 256, self.layers[3][0], stride=2)
        self.layer32 = self._make_layer_res3D(self.resblock3D, 256, 64, self.layers[3][1])

        self.out_CNN = nn.Sequential(
                                    nn.Conv3d(256, self.emb_size, 1),
                                    nn.AvgPool3d((1,3,3), stride=1)
                                )
        # self.out_CNN = nn.Conv3d(256, self.emb_size, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,0,0))


    def _make_layer_res2D(self, resblock2D, planes, blocks, stride):
        downsample = None
        if stride!=1 or self.inplanes!=planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=(1,stride,stride)),
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(resblock2D(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(resblock2D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_layer_res3D(self, resblock3D, planes, inter_channels, blocks):
        downsample = None
        layers = []
        for i in range(blocks):
            layers.append(resblock3D(planes, planes, inter_channels))
        return nn.Sequential(*layers)
    

    def forward(self,x):

        # Preproces inpute shape and dimension sequence
        if len(x.shape)==4:
            x = x[:,None,:,:,:]

        x = self.inpConv(x)
        
        x = self.layer01(x)
        x = self.layer02(x)
        
        x = self.layer11(x)
        x = self.layer12(x)

        x = self.upsample1(x)

        x = self.layer21(x)
        x = self.layer22(x)

        x = self.upsample2(x)
        
        x = self.layer31(x)
        x = self.layer32(x)

        x = self.out_CNN(x)

        B, S, N, _, _ = x.shape
        x = torch.reshape(x, (B,S,N)).permute(0,2,1)

        return x
