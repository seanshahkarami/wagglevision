import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet101 as _fcn_resnet101


class UnwrappedFCN(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)['out']


def fcn_resnet101(pretrained=False, num_classes=2):
    if pretrained is False:
        return UnwrappedFCN(_fcn_resnet101(pretrained=False, num_classes=num_classes))

    net = _fcn_resnet101(pretrained=True)

    # replace last layer with num_classes conv layer
    children = dict(net.named_children())
    children['classifier'][-1] = nn.Conv2d(
        in_channels=512,
        out_channels=num_classes,
        kernel_size=1,
        stride=1)

    return UnwrappedFCN(net)
