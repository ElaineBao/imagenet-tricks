import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['SENet', 'senet154', 'se_resnetd50', 'se_resnetd101', 'se_resnetd152',
           'se_resnextd50_32x4d', 'se_resnextd50_64x4d',
           'se_resnextd101_32x4d', 'se_resnextd101_64x4d',
           'se_resnextd152_32x4d','se_resnextd152_64x4d']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se_module(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = planes * base_width * groups // 64
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 conv1_filter=64, inplanes=128, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        conv1_filter (int):  Number of output channels for conv1.
            - For SENet154: 64
            - For SE-ResNet models: 32
            - For SE-ResNeXt models: 32
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_filter, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv1_filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv1_filter, conv1_filter, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv1_filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv1_filter, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction,
                        downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], groups=groups, reduction=reduction,
                        downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding, stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], groups=groups, reduction=reduction,
                        downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding, stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], groups=groups, reduction=reduction,
                        downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if downsample_kernel_size > stride: # no information can lost when conv
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=downsample_kernel_size, stride=stride,
                              padding=downsample_padding, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else: # information can lost if no avgpool
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=downsample_kernel_size,
                              padding=downsample_padding, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def se_resnetd50(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnetd50']))
    return model


def se_resnetd101(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnetd101']))
    return model


def se_resnetd152(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnetd152']))
    return model


def se_resnextd50_32x4d(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnetd50_32x4d']))
    return model

def se_resnextd50_64x4d(num_classes=1000, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=64, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnetd50_64x4d']))
    return model

def se_resnextd101_32x4d(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnextd101_32x4d']))
    return model

def se_resnextd101_64x4d(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=64, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnextd101_64x4d']))
    return model

def se_resnextd152_32x4d(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 8, 36, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnextd101_32x4d']))
    return model

def se_resnextd152_64x4d(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=None, inplanes=64, conv1_filter=32,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnextd101_64x4d']))
    return model

def senet154(num_classes=1000, pretrained=False, **kwargs):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, inplanes=128, conv1_filter=64,
                  downsample_kernel_size=3, downsample_padding=1,
                  num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['senet154']))
    return model

if __name__ == '__main__':
    import torch
    model = se_resnetd152(num_classes=400)
    data = torch.randn((1,3,224,224))
    forward = model(data)
    print(forward, forward.size())
