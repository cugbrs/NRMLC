import torch.nn as nn
from torchvision import models
import torch
import math
from DecoupledCrossAttentionModule import DecoupledCrossAttentionWithPE
from TransformerBasedMiltiClassification import TransformerClassifierHead


def model(class_num, pretrained, requires_grad):

    model = NRMLC(class_num=class_num, pretrained=pretrained)

    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False

    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True

    return model


class NRMLC(nn.Module):
    def __init__(self, class_num, pretrained):
        super(NRMLC, self).__init__()
        self.net = models.resnet50(pretrained=pretrained)
        self.class_num = class_num
        self.linear = nn.Linear(2048, self.class_num)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=1, stride=4)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=2)
        self.conv3 = nn.Conv2d(2048, 1024, kernel_size=1)

        self.LEWA = DecoupledCrossAttentionWithPE(
            num_classes=self.class_num,
            feature_dim=2048,
            num_heads=8,
            normalize=True,
            scale=2 * math.pi
        )

        self.TBML = TransformerClassifierHead(
            feature_dim=2048,
            num_heads=8,
            ffn_hidden_dim=4096,
            dropout_rate=0.2
        )

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        # MSFE
        self.feat1 = self.conv1(output)
        output = self.net.layer3(output)
        self.feat2 = self.conv2(output)
        output = self.net.layer4(output)
        self.feat3 = self.conv3(output)
        self.confeat = torch.cat((self.feat1, self.feat2, self.feat3), dim=1)
        self.feat = self.net.avgpool(self.confeat)
        # LWEA
        enhanced_features, attention_weights = self.LEWA(self.confeat)  # [B, C, D]
        self.attention_weights = attention_weights
        # TBMC
        output = self.TBML(enhanced_features)  # outpput need [B, C, D]; output without activation function
        return output
