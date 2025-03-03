import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer


# Image feature extractor
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512, explain_model=False):
        super().__init__()
        self.explain_model = explain_model
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        if self.explain_model:
            x = self.feature_extractor(x)
        else:
            with torch.no_grad():
                x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Text feature extractor
class BertFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512, explain_model=False):
        super().__init__()
        self.explain_model = explain_model
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        if self.explain_model:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_embedding)


class FusionModel(nn.Module):
    def __init__(self, latent_dim=512, num_classes=2, explain_model=False):
        super().__init__()

        self.image_encoder = ResNetFeatureExtractor(output_dim=latent_dim, explain_model=explain_model)
        self.text_encoder = BertFeatureExtractor(output_dim=latent_dim, explain_model=explain_model)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 4),
            nn.Linear(latent_dim // 4, latent_dim)
        )

        self.mlp_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, image, text_input_ids, text_attention_mask):
        image_features = self.image_encoder(image) # [batch_size, 1, 512]
        text_features = self.text_encoder(text_input_ids, text_attention_mask) # [batch_size, 1, 512]

        x = torch.cat([image_features, text_features], dim=-1)  # [batch_size, 1024]

        x = self.fusion_mlp(x)
        x = self.mlp_classifier(x)

        return x # [batch_size, num_classes]
