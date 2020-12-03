# loosely inspired by https://github.com/chengyangfu/pytorch-vgg-cifar10
import math
import torch.nn as nn

from layers import RealOpu, SimulatedOpu, Sign, Identity
from tinydfa import DFA, DFALayer


class CifarVGG16(nn.Module):

    def __init__(self, opu, binary_layer, opu_features, features, opu_input, sign_back, real_opu, device, dataset):
        super(CifarVGG16, self).__init__()
        self.features = features
        if dataset == 'cifar10':
            self.number_classes = 10
        if dataset == 'cifar100':
            self.number_classes = 100

        if binary_layer:
            if sign_back:
                print(f'- Binary layer = {binary_layer} but sign_back = {sign_back}--> attack method = FA!')
                self.dfa1 = Identity()
                self.dfa = Identity()
            else:
                self.training_method = 'DFA'
                print(f'- Binary layer = {binary_layer} --> training method = DFA!')
                self.dfa1 = DFALayer()
                self.dfa = DFA([self.dfa1], no_training=(self.training_method == 'SHALLOW'))
        else:
            if opu:
                print(f'- Binary layer = {binary_layer}, opu = {opu} --> training method = FA')
            else:
                print(f'- Binary layer = {binary_layer}, opu = {opu} --> training method = BP')
            self.dfa1 = nn.Dropout()
            self.dfa = Identity()

        if opu:
            print(f'- opu_input = {opu_input}, opu_output = {opu_features}')
            if real_opu:
                print('- VGG with real OPU')
                print(opu_input, opu_features)
                second_classifier_layer = RealOpu(in_features=opu_input, out_features=opu_features, device=device)
            else:
                print('- VGG with simulated OPU')
                second_classifier_layer = SimulatedOpu(in_features=opu_input, out_features=opu_features)

        else:
            print('- VGG without OPU')
            second_classifier_layer = nn.Linear(opu_input, opu_features)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, opu_input),
            nn.ReLU(True),
            self.dfa1,
            Sign(binary_layer, sign_back=sign_back),
            second_classifier_layer,
            nn.ReLU(True),
            nn.Linear(opu_features, self.number_classes),
            self.dfa
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AblationCifarVGG16(nn.Module):

    def __init__(self, opu, binary_layer, opu_features, features, opu_input, dataset, sign_back):
        super(AblationCifarVGG16, self).__init__()
        self.features = features
        print(f'OPU: {opu}, binary_layer: {binary_layer}, oo: {opu_features}, oi: {opu_input}, sign back: {sign_back}')

        if dataset == 'cifar10':
            self.number_classes = 10
        if dataset == 'cifar100':
            self.number_classes = 100

        if sign_back:
            self.dfa1 = Identity()
            self.dfa = Identity()
        else:
            self.training_method = 'DFA'
            self.dfa1 = DFALayer()
            self.dfa = DFA([self.dfa1], no_training=(self.training_method == 'SHALLOW'))

        if opu:
            second_classifier_layer = SimulatedOpu(in_features=opu_input, out_features=opu_features)
        else:
            second_classifier_layer = nn.Linear(opu_input, opu_features)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, opu_input),
            nn.ReLU(True),
            self.dfa1,
            Sign(binary_layer, sign_back=sign_back),
            second_classifier_layer,
            nn.ReLU(True),
            nn.Linear(opu_features, self.number_classes),
            self.dfa
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def cifar_vgg16(opu, binary_layer, opu_features, opu_input, sign_back, device, dataset, real_opu=False):
    return CifarVGG16(opu=opu, binary_layer=binary_layer, opu_features=opu_features, features=make_layers(cfg),
                      opu_input=opu_input, sign_back=sign_back, real_opu=real_opu, device=device, dataset=dataset)


def ablation_cifar_vgg16(opu, binary_layer, opu_features, opu_input, dataset, sign_back=False):
    return AblationCifarVGG16(opu=opu, binary_layer=binary_layer, opu_features=opu_features, features=make_layers(cfg),
                              opu_input=opu_input, dataset=dataset, sign_back=sign_back)


