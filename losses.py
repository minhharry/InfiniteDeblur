import torch
import torch.nn as nn
import lpips
import torch.nn.functional as F
from torchvision.transforms import v2
import torchvision

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load the pre-trained VGG model
        self.vgg = torchvision.models.vgg16(weights='DEFAULT').features[:23]
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.loss_fn = nn.MSELoss()  
        self.layers = [3, 8, 15, 22]  
        
    def forward(self, generated, target):
        loss = 0
        for i, layer in enumerate(self.vgg):
            generated = layer(generated)
            target = layer(target)
            if i in self.layers:
                loss += self.loss_fn(generated, target)
        return loss

class RandomPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        """
            VGG-like perceptual loss with the first Conv2d layer randomized each iteration.
        """
        super().__init__()
        SCALE = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 8*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(8*SCALE, 16*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(16*SCALE, 32*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net4 = nn.Sequential(
            nn.Conv2d(32*SCALE, 64*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False
        for param in self.net3.parameters():
            param.requires_grad = False
        for param in self.net4.parameters():
            param.requires_grad = False

        self.loss_fn = nn.L1Loss()
    def reset(self):
        self.net.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)

    def forward(self, input, target):
        self.reset()
        input = self.net(input)
        target = self.net(target)
        net_loss = self.loss_fn(input, target)
        input = self.net2(input)
        target = self.net2(target)
        net_loss += self.loss_fn(input, target) * 2
        input = self.net3(input)
        target = self.net3(target)
        net_loss += self.loss_fn(input, target) * 4
        input = self.net4(input)
        target = self.net4(target)
        net_loss += self.loss_fn(input, target) * 8
        return net_loss
    
class RandomPerceptualLossResetAll(nn.Module):
    def __init__(self) -> None:
        """
            VGG-like perceptual loss with all Conv2d layers randomized each iteration.
        """
        super().__init__()
        SCALE = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 8*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(8*SCALE, 16*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(16*SCALE, 32*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net4 = nn.Sequential(
            nn.Conv2d(32*SCALE, 64*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False
        for param in self.net3.parameters():
            param.requires_grad = False
        for param in self.net4.parameters():
            param.requires_grad = False

        self.loss_fn = nn.L1Loss()
    def reset(self):
        self.net.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)
        self.net2.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)
        self.net3.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)
        self.net4.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)

    def forward(self, input, target):
        self.reset()
        input = self.net(input)
        target = self.net(target)
        net_loss = self.loss_fn(input, target)
        input = self.net2(input)
        target = self.net2(target)
        net_loss += self.loss_fn(input, target) * 2
        input = self.net3(input)
        target = self.net3(target)
        net_loss += self.loss_fn(input, target) * 4
        input = self.net4(input)
        target = self.net4(target)
        net_loss += self.loss_fn(input, target) * 8
        return net_loss

class RandomPerceptualLossNoReset(nn.Module):
    def __init__(self) -> None:
        """
            VGG-like perceptual loss with all Conv2d layers randomized.
        """
        super().__init__()
        SCALE = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 8*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(8*SCALE, 16*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(16*SCALE, 32*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net4 = nn.Sequential(
            nn.Conv2d(32*SCALE, 64*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False
        for param in self.net3.parameters():
            param.requires_grad = False
        for param in self.net4.parameters():
            param.requires_grad = False

        self.loss_fn = nn.L1Loss()

    def forward(self, input, target):
        input = self.net(input)
        target = self.net(target)
        net_loss = self.loss_fn(input, target)
        input = self.net2(input)
        target = self.net2(target)
        net_loss += self.loss_fn(input, target) * 2
        input = self.net3(input)
        target = self.net3(target)
        net_loss += self.loss_fn(input, target) * 4
        input = self.net4(input)
        target = self.net4(target)
        net_loss += self.loss_fn(input, target) * 8
        return net_loss

class RandomPerceptualLossLastLayerNoMaxPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        SCALE = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 8*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(8*SCALE, 16*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(16*SCALE, 32*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.net4 = nn.Sequential(
            nn.Conv2d(32*SCALE, 64*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False
        for param in self.net3.parameters():
            param.requires_grad = False
        for param in self.net4.parameters():
            param.requires_grad = False

        self.loss_fn = nn.L1Loss()
    def reset(self):
        self.net.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)

    def forward(self, input, target):
        self.reset()
        input = self.net(input)
        target = self.net(target)
        input = self.net2(input)
        target = self.net2(target)
        input = self.net3(input)
        target = self.net3(target)
        input = self.net4(input)
        target = self.net4(target)
        net_loss = self.loss_fn(input, target)
        return net_loss

class RandomPerceptualLossLastLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        SCALE = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 8*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(8*SCALE, 16*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(16*SCALE, 32*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.net4 = nn.Sequential(
            nn.Conv2d(32*SCALE, 64*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False
        for param in self.net3.parameters():
            param.requires_grad = False
        for param in self.net4.parameters():
            param.requires_grad = False

        self.loss_fn = nn.L1Loss()
    def reset(self):
        self.net.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)

    def forward(self, input, target):
        self.reset()
        input = self.net(input)
        target = self.net(target)
        input = self.net2(input)
        target = self.net2(target)
        input = self.net3(input)
        target = self.net3(target)
        input = self.net4(input)
        target = self.net4(target)
        net_loss = self.loss_fn(input, target) 
        return net_loss

class RandomPerceptualLossLastLayerResetAll(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        SCALE = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 8*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(8*SCALE, 16*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(16*SCALE, 32*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32*SCALE, 64*SCALE, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        for param in self.net.parameters():
            param.requires_grad = False

        self.loss_fn = nn.L1Loss()
    def reset(self):
        self.net.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)

    def forward(self, input, target):
        self.reset()
        input = self.net(input)
        target = self.net(target)
        net_loss = self.loss_fn(input, target) 
        return net_loss

class RandomPerceptualLossLastLayerNoReset(nn.Module):
    def __init__(self, scale=1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8*scale, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(8*scale, 16*scale, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(16*scale, 32*scale, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32*scale, 64*scale, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        for param in self.net.parameters():
            param.requires_grad = False

        self.loss_fn = nn.L1Loss()

    def reset(self):
        self.net.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)

    def forward(self, input, target):
        input = self.net(input)
        target = self.net(target)
        net_loss = self.loss_fn(input, target) 
        return net_loss

class ResizeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
        self.resize1 = v2.Resize(128)
        self.resize2 = v2.Resize(64)
        self.resize3 = v2.Resize(32)
        self.resize4 = v2.Resize(16)

    def forward(self, input, target):
        net_loss = self.loss_fn(input, target)
        input = self.resize1(input)
        target = self.resize1(target)
        net_loss += self.loss_fn(input, target) * 2
        input = self.resize2(input)
        target = self.resize2(target)
        net_loss += self.loss_fn(input, target) * 4
        input = self.resize3(input)
        target = self.resize3(target)
        net_loss += self.loss_fn(input, target) * 8
        input = self.resize4(input)
        target = self.resize4(target)
        net_loss += self.loss_fn(input, target) * 16
        return net_loss

class PSNRloss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return - 10 * torch.log10((torch.max(target) - torch.min(target))**2 / F.mse_loss(pred, target))
    
class LPIPSLoss(nn.Module):
    def __init__(self, net='alex'):
        super().__init__()
        self.loss = lpips.LPIPS(net=net)
        for param in self.loss.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # Normalize input and target to [-1, 1]
        input = torch.clamp(input, 0, 1)
        target = torch.clamp(target, 0, 1)
        input = input * 2 - 1
        target = target * 2 - 1
        
        return self.loss(input, target).squeeze().mean()
    
if __name__ == '__main__':
    from torchinfo import summary

    model = RandomPerceptualLoss()
    summary(model, ((16, 3, 256, 256),(16, 3, 256, 256)))