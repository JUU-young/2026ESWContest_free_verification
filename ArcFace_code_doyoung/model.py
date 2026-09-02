import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from torchvision.models import (
    MobileNet_V3_Small_Weights
)


class SignatureBackbone(nn.Module):


    def __init__(
        self,
        embedding_size=128,
        pretrained=True
    ):

        super(
            SignatureBackbone,
            self
        ).__init__()


        if pretrained:

            weights = (
                MobileNet_V3_Small_Weights.DEFAULT
            )

        else:

            weights = None


        self.backbone = (
            models.mobilenet_v3_small(
                weights=weights
            )
        )


        in_features = (
            self.backbone
            .classifier[0]
            .in_features
        )


        self.backbone.classifier = (
            nn.Sequential(

                nn.Linear(
                    in_features,
                    embedding_size
                ),

                nn.BatchNorm1d(
                    embedding_size
                )
            )
        )


    def forward(
        self,
        x
    ):


        features = self.backbone(
            x
        )


        features = F.normalize(
            features,
            p=2,
            dim=1
        )


        return features


class ArcFaceLoss(nn.Module):

    """
    ArcFace Angular Margin Loss

    embeddings:
        [B, embedding_size]

    labels:
        [B]
    """


    def __init__(
        self,
        num_classes,
        embedding_size=128,
        s=30.0,
        m=0.5
    ):

        super(
            ArcFaceLoss,
            self
        ).__init__()


        self.s = float(s)
        self.m = float(m)


        self.weight = nn.Parameter(

            torch.empty(
                num_classes,
                embedding_size
            )

        )


        nn.init.xavier_uniform_(
            self.weight
        )


        self.register_buffer(

            "cos_m",

            torch.tensor(
                math.cos(
                    self.m
                ),
                dtype=torch.float32
            )
        )


        self.register_buffer(

            "sin_m",

            torch.tensor(
                math.sin(
                    self.m
                ),
                dtype=torch.float32
            )
        )


    def forward(
        self,
        embeddings,
        labels
    ):

        normalized_weight = (
            F.normalize(
                self.weight,
                p=2,
                dim=1
            )
        )

        cosine = F.linear(
            embeddings,
            normalized_weight
        )


        cosine = cosine.clamp(
            -1.0 + 1e-7,
            1.0 - 1e-7
        )


        sine = torch.sqrt(

            torch.clamp(

                1.0
                -
                cosine.pow(2),

                min=0.0
            )

        )


        phi = (
            cosine
            *
            self.cos_m

            -

            sine
            *
            self.sin_m
        )


        one_hot = torch.zeros_like(
            cosine
        )


        one_hot.scatter_(
            1,
            labels.view(
                -1,
                1
            ).long(),
            1.0
        )


        output = (

            one_hot
            *
            phi

            +

            (
                1.0
                -
                one_hot
            )
            *
            cosine
        )

        output = (
            output
            *
            self.s
        )

        loss = F.cross_entropy(
            output,
            labels
        )


        return loss