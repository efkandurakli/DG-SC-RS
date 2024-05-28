import random
import torch
import torch.nn as nn
from resnet import ResNet
from torchvision.models.resnet import BasicBlock


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        batch_size = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sigma = (var + self.eps).sqrt()
        mu, sigma = mu.detach(), sigma.detach()
        x_normed = (x - mu) / sigma

        interpolation = self.beta.sample((batch_size, 1, 1, 1))
        interpolation = interpolation.to(x.device)

        # split into two halves and swap the order
        perm = torch.arange(batch_size - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(batch_size // 2)]
        perm_a = perm_a[torch.randperm(batch_size // 2)]
        perm = torch.cat([perm_b, perm_a], 0)

        mu_perm, sigma_perm = mu[perm], sigma[perm]
        mu_mix = mu * interpolation + mu_perm * (1 - interpolation)
        sigma_mix = sigma * interpolation + sigma_perm * (1 - interpolation)

        return x_normed * sigma_mix + mu_mix
    


def _resnet_with_mix_style(block, layers, progress, mix_layers=None, mix_p=0.5, mix_alpha=0.1,
                           resnet_class=ResNet, **kwargs):

    if mix_layers is None:
        mix_layers = []

    class ResNetWithMixStyleModule(resnet_class):
        def __init__(self, mix_layers, mix_p=0.5, mix_alpha=0.1, *args, **kwargs):
            super(ResNetWithMixStyleModule, self).__init__(*args, **kwargs)
            self.mixStyleModule = MixStyle(p=mix_p, alpha=mix_alpha)
            for layer in mix_layers:
                assert layer in ['layer1', 'layer2', 'layer3']
            self.apply_layers = mix_layers

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
   
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            if 'layer1' in self.apply_layers:
                x = self.mixStyleModule(x)
            x = self.layer2(x)
            if 'layer2' in self.apply_layers:
                x = self.mixStyleModule(x)
            x = self.layer3(x)
            if 'layer3' in self.apply_layers:
                x = self.mixStyleModule(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

    model = ResNetWithMixStyleModule(mix_layers=mix_layers, mix_p=mix_p, mix_alpha=mix_alpha, block=block,
                                     layers=layers, **kwargs)

    return model


def resnet18(progress=True, **kwargs):
    return _resnet_with_mix_style(BasicBlock, [2, 2, 2, 2], progress, **kwargs)