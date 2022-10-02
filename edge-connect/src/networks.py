import torch
import torchvision
import torch.nn as nn

from .network_module import *

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(5, 48, 5, 1, 2, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48, 48 * 2, 3, 2, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 2, 48 * 2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 2, 48 * 4, 3, 2, 1, pad_type = "zero", activation = "elu", norm = "none"),
            # Bottleneck
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 2, dilation = 2, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 4, dilation = 4, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 8, dilation = 8, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 16, dilation = 16, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            # decoder
            TransposeGatedConv2d(48 * 4, 48 * 2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 2, 48 * 2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            TransposeGatedConv2d(48 * 2, 48, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48, 48//2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48//2, 3, 3, 1, 1, pad_type = "zero", activation = 'none', norm = "none"),
            nn.Tanh()
      )
        
        self.refine_conv = nn.Sequential(
            GatedConv2d(5, 48, 5, 1, 2, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48, 48, 3, 2, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48, 48*2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*2, 48*2, 3, 2, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*2, 48*4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*4, 48*4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 2, dilation = 2, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 4, dilation = 4, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 8, dilation = 8, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48 * 4, 48 * 4, 3, 1, 16, dilation = 16, pad_type = "zero", activation = "elu", norm = "none")
        )
        self.refine_atten_1 = nn.Sequential(
            GatedConv2d(5, 48, 5, 1, 2, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48, 48, 3, 2, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48, 48*2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*2, 48*4, 3, 2, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*4, 48*4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*4, 48*4, 3, 1, 1, pad_type = "zero", activation = 'relu', norm = "none")
        )
        self.refine_atten_2 = nn.Sequential(
            GatedConv2d(48*4, 48*4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*4, 48*4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none")
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(48*8, 48*4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*4, 48*4, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            TransposeGatedConv2d(48 * 4, 48*2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48*2, 48*2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            TransposeGatedConv2d(48 * 2, 48, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48, 48//2, 3, 1, 1, pad_type = "zero", activation = "elu", norm = "none"),
            GatedConv2d(48//2, 3, 3, 1, 1, pad_type = "zero", activation = 'none', norm = "none"),
            nn.Tanh()
        )
        self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                     fuse=True)
        if init_weights:
            self.init_weights(init_type='kaiming')
        
    def forward(self, img, edge, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse
        # import pdb; pdb.set_trace()
        # print(img.shape)
        # print(mask.shape)
        # print(edge.shape)
        first_masked_img = img * (1 - mask) + mask
        # print(first_masked_img.shape)

        first_in = torch.cat((first_masked_img, edge, mask), dim=1)       # in: [B, 4, H, W]
        # print(first_in.shape)
        first_out = self.coarse(first_in)                           # out: [B, 3, H, W]
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, edge, mask], dim=1)
        refine_conv = self.refine_conv(second_in)     
        refine_atten = self.refine_atten_1(second_in)
        mask_s = nn.functional.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)
        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))
        return first_out, second_out

#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(5, 48, 7, 1, 3, pad_type = "zero", activation = "elu", norm = "none", sn = True)
        self.block2 = Conv2dLayer(48, 48 * 2, 4, 2, 1, pad_type = "zero", activation = "elu", norm = "none", sn = True)
        self.block3 = Conv2dLayer(48 * 2, 48 * 4, 4, 2, 1, pad_type = "zero", activation = "elu", norm = "none", sn = True)
        self.block4 = Conv2dLayer(48 * 4, 48 * 4, 4, 2, 1, pad_type = "zero", activation = "elu", norm = "none", sn = True)
        self.block5 = Conv2dLayer(48 * 4, 48 * 4, 4, 2, 1, pad_type = "zero", activation = "elu", norm = "none", sn = True)
        self.block6 = Conv2dLayer(48 * 4, 1, 4, 2, 1, pad_type = "zero", activation = 'none', norm = 'none', sn = True)

        if init_weights:
            self.init_weights()

    def forward(self, img, edge, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask, edge), 1)
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(BaseNetwork):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
