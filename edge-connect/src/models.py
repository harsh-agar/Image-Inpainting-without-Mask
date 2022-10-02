import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .networks import GatedGenerator, PatchDiscriminator, EdgeGenerator, Discriminator, PerceptualNet
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = GatedGenerator()
        perceptualnet = PerceptualNet()
        discriminator = PatchDiscriminator()
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        # perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptualnet', perceptualnet)
        # self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1
        
        # Tensor type
        Tensor = torch.cuda.FloatTensor

        gen_loss = 0
        dis_loss = 0
        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        valid = Tensor(np.ones((images.shape[0], 1, images.shape[2]//32, images.shape[3]//32)))
        fake  = Tensor(np.zeros((images.shape[0], 1, images.shape[2]//32, images.shape[3]//32)))
        zero  = Tensor(np.zeros((images.shape[0], 1, images.shape[2]//32, images.shape[3]//32)))

        # process outputs
        first_out, second_out = self(images, edges, masks)

        # forward propagation
        first_out_wholeimg  = images * (1 - masks) + first_out * masks        # in range [0, 1]
        second_out_wholeimg = images * (1 - masks) + second_out * masks      # in range [0, 1]

        # discriminator loss
        fake_scalar = self.discriminator(second_out_wholeimg.detach(), edges, masks)
        true_scalar = self.discriminator(images, edges, masks)
        loss_fake = -torch.mean(torch.min(zero, - valid - fake_scalar))
        loss_true = -torch.mean(torch.min(zero, - valid + true_scalar))
        # Overall Loss and optimize
        dis_loss = 0.5 * (loss_fake + loss_true)

        # dis_real_loss = self.adversarial_loss(true_scalar, True, True)
        # dis_fake_loss = self.adversarial_loss(fake_scalar, False, True)
        # dis_loss += (dis_real_loss + dis_fake_loss) / 2
        
        # L1 Loss
        gen_first_l1_loss  = (first_out  - images).abs().mean()
        gen_second_l1_loss = (second_out - images).abs().mean()

        # generator l1 loss
        # gen_first_l1_loss  = self.l1_loss(first_out_wholeimg,  images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        # gen_second_l1_loss = self.l1_loss(second_out_wholeimg, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)

        gen_loss += (gen_first_l1_loss + gen_second_l1_loss) * self.config.L1_LOSS_WEIGHT_INPAINT

        #GAN Loss
        fake_scalar = self.discriminator(second_out_wholeimg, masks, edges)
        gen_gan_loss = -torch.mean(fake_scalar)

        # # generator adversarial loss
        # gen_input_fake = second_out_wholeimg
        # gen_fake  = self.discriminator(second_out_wholeimg.detach(), edges, masks)                    # in: [rgb(3)]
        # gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT

        # Get the deep semantic feature maps, and compute Perceptual Loss
        img_featuremaps = self.perceptualnet(images)                          # feature maps
        second_out_featuremaps = self.perceptualnet(second_out_wholeimg.detach())
        second_PerceptualLoss = self.l1_loss(second_out_featuremaps, img_featuremaps) 

        gen_loss += second_PerceptualLoss * self.config.CONTENT_LOSS_WEIGHT_INPAINT

        # generator style loss
        gen_style_loss = self.style_loss(second_out_wholeimg.detach() * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        # gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2",        dis_loss.item()),
            ("l_g2",        gen_gan_loss.item()),
            ("first_l_l1",  gen_first_l1_loss.item()),
            ("second_l_l1", gen_second_l1_loss.item()),
            ("l_per",       second_PerceptualLoss.item()),
            ("l_sty",       gen_style_loss.item()),
        ]

        return second_out, second_out_wholeimg, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        # images_masked = (images * (1 - masks).float()) + masks
        # inputs = torch.cat((images_masked, edges), dim=1)
        first_out, second_out = self.generator(images, edges, masks)                                    # in: [rgb(3) + edge(1)]
        return first_out, second_out

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()
