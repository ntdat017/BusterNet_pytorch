import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils import Conv2dStaticSamePadding as Conv2d

class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    '''BatchNorm Inception module with batch normalization
    Input:
        x = tensor4D, (n_samples, n_rows, n_cols, n_feats)
    '''
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, conv_block=None, is_last=False):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2D
        if is_last:
            k_size1,  k_size2, k_size3 = 5, 7, 11
        else:
            k_size1,  k_size2, k_size3 = 1, 3, 5
        
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=k_size1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=k_size2, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=k_size3, padding=1)
        )
    
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        outputs = torch.cat((x1, x2, x3), dim=1)
        return outputs

class CorrelationPercPooling(nn.Module):
    '''Custom Self-Correlation Percentile Pooling Layer
    '''
    def __init__(self, nb_pools=256, **kwargs):
        super(CorrelationPercPooling, self).__init__()
        self.nb_pools = nb_pools

        n_maps = 16*16
        
        if self.nb_pools is not None:
            self.ranks = torch.floor(torch.linspace(0, n_maps -1, self.nb_pools)).type(torch.long)
        else:
            self.ranks = torch.range(1, n_maps, dtype=torch.long)

    def forward(self, x):
        '''
            x_shape: (n, c, h, w)
        '''
        n_bsize, n_feats, n_cols, n_rows = x.shape
        n_maps = n_cols * n_rows
        x_3d = x.reshape(n_bsize, n_feats, n_maps)

        x_corr_3d = torch.matmul(x_3d.transpose(1, 2), x_3d) / n_feats
        x_corr = x_corr_3d.reshape(n_bsize, n_maps, n_cols, n_rows)

        # ranks = ranks.to(devices)
        x_sort, _ = torch.topk(x_corr, k=n_maps, dim=1, sorted=True)

        x_f1st_sort = x_sort.permute(1, 2, 3, 0)
        x_f1st_pool = x_f1st_sort[self.ranks]
        x_pool = x_f1st_pool.permute(3, 0, 1, 2)

        return x_pool 

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        h_channels = out_channels // 2
        self.inception = Inception(in_channels, out_channels, h_channels, out_channels, h_channels, out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.inception(x)
        return x

class MaskDecoder(nn.Module):
    def __init__(self, in_channels=512):
        super(MaskDecoder, self).__init__()
        self.f16 = Inception(in_channels, 8, 4, 8, 4, 8)

        self.deconv_0 = DeconvBlock(24, 6)
        self.deconv_1 = DeconvBlock(18, 4)
        self.deconv_2 = DeconvBlock(12, 2)
        self.deconv_3 = DeconvBlock(6, 2)

        self.pred_mask = Inception(6, 2, 1, 2, 1, 2, is_last=True)

    def forward(self, x):
        f16 = self.f16(x)
        f32 = self.deconv_0(f16)
        f64 = self.deconv_1(f32)
        f128 = self.deconv_2(f64)
        f256 = self.deconv_3(f128)
        pred_mask = self.pred_mask(f256)

        return pred_mask

class ManipulationNet(nn.Module):
    def __init__(self):
        super(ManipulationNet, self).__init__()
        # self.features = nn.Sequential(*make_layers(cfgs['C']))
        self.features = models.vgg16_bn().features[:-10]
        self.mask_decoder = MaskDecoder(512)
        self.classifier = nn.Sequential(
            Conv2d(6, 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.mask_decoder(x)
        mask = self.classifier(x)
        return x, mask

class SimilarityNet(nn.Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()
        # self.features = nn.Sequential(*make_layers(cfgs['C']))
        self.features = models.vgg16_bn().features[:-10]
        self.correlation_per_pooling = CorrelationPercPooling(nb_pools=256)
        self.mask_decoder = MaskDecoder(256)
        self.classifier = nn.Sequential(
            Conv2d(6, 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.correlation_per_pooling(x)
        x = self.mask_decoder(x)
        mask = self.classifier(x)
        return x, mask

class BusterNet(nn.Module):
    def __init__(self, image_size):
        super(BusterNet, self).__init__()

        self.image_size = image_size
        
        self.manipulation_net = ManipulationNet()
        self.similarity_net = SimilarityNet()

        self.inception = nn.Sequential(
            Inception(12, 3, 3, 3, 3, 3),
            Conv2d(9, 3, kernel_size=3),
            nn.Softmax2d()
        )

    def forward(self, x):
        mani_feat, mani_output = self.manipulation_net(x)
        simi_feat, simi_output = self.similarity_net(x)

        merged_feat = torch.cat([simi_feat, mani_feat], dim=1)

        x = self.inception(merged_feat)

        mask_out = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear')
        return mask_out, mani_output, simi_output

if __name__ == "__main__":
    model = BusterNet(256)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
