import torch
import torch.nn as nn
from torchvision import models
from monai.networks.nets import DenseNet121


class Encoder2D(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, emb_dim)

    def forward(self, x):
        return self.model(x)

# --- NUOVO ENCODER 3D MEDICO ---
class Encoder3D(nn.Module):
    """Estrae feature 3D usando l'architettura SOTA DenseNet121 di MONAI"""
    def __init__(self, emb_dim=64):
        super().__init__()
        # spatial_dims=3 indica che stiamo lavorando su volumi, non immagini piatte
        self.model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=emb_dim)

    def forward(self, x):
        return self.model(x)


class EncoderTab(nn.Module):
    """Estrae feature dai dati clinici (età, test, ecc.)"""

    def __init__(self, input_dim=8, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, emb_dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Fonde le informazioni usando Multi-Head Attention"""

    def __init__(self, emb_dim=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim)
        )

    def forward(self, query, key_value):
        attn_output, _ = self.attn(query, key_value, key_value)
        out = self.norm(query + attn_output)
        return (out + self.ffn(out)).squeeze(1)


class KFoldAttentionModel(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.enc_2d = Encoder2D(emb_dim)
        self.enc_3d = Encoder3D(emb_dim)
        self.enc_tab = EncoderTab(8, emb_dim)
        self.fusion = CrossAttentionFusion(emb_dim=emb_dim)

        # Aumentiamo il dropout nel classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Più alto per evitare l'overfitting sui dati clinici
            nn.Linear(64, 3)
        )

    def forward(self, x_tab, x_2d, x_3d, force_images=False):
        e_tab = self.enc_tab(x_tab).unsqueeze(1)
        e_2d = self.enc_2d(x_2d).unsqueeze(1)
        e_3d = self.enc_3d(x_3d).unsqueeze(1)

        # Se siamo in training, ogni tanto azzeriamo i dati tabellari
        # per forzare il modello a guardare le immagini
        if self.training and torch.rand(1) < 0.2:  # 20% delle volte ignora la tabella
            e_tab = e_tab * 0

        e_images = torch.cat([e_2d, e_3d], dim=1)
        fused = self.fusion(e_tab, e_images)
        return self.classifier(fused)