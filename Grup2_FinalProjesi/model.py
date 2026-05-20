"""
model.py – Hiyerarşik İHA Navigasyon Modeli
=============================================
Üst Katman  : VAE  (Encoder-Decoder, waypoint gizil vektörü z_w)
Orta Katman : MulVPRL  (ResNet8 shared-weights encoder + GRU)
Alt Katman  : MLP Heads  (steering regresyon + collision sınıflandırma)

Toplam parametre ≈ 2.01M
  Encoder (ResNet8)  : ~1.27M
  VAE fc_mu/logvar   : ~0.03M
  VAE Decoder (küçük): ~0.41M
  GRU (hidden=200)   : ~0.22M
  MLP Heads (x2)     : ~0.02M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ResNet8 yapı taşı ─────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


# ── Paylaşımlı ResNet8 Encoder ────────────────────────────────────────
class ResNet8Encoder(nn.Module):
    """Giriş: [B,1,H,W]  →  Çıkış: [B, feat_dim]"""
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ResBlock(32,  64,  stride=2)
        self.layer2 = ResBlock(64,  128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.proj   = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.proj(self.pool(x).flatten(1))


# ── VAE ───────────────────────────────────────────────────────────────
class VAEEncoder(nn.Module):
    """[B,T,1,H,W] → mu, log_var  her biri [B, latent_dim]"""
    def __init__(self, frame_enc: ResNet8Encoder,
                 feat_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.frame_enc = frame_enc
        self.fc_mu     = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.frame_enc(x.view(B*T, C, H, W)).view(B, T, -1).mean(1)
        return self.fc_mu(feats), self.fc_logvar(feats)


class VAEDecoder(nn.Module):
    """
    Hafif decoder: latent_dim → [B,1,H,W]
    Kanal dizisi: 64 → 32 → 16 → 8 → 4 → 1
    """
    def __init__(self, latent_dim: int = 64, img_h: int = 244, img_w: int = 324):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.fc = nn.Linear(latent_dim, 64 * 8 * 11)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,  8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 8,  4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 4,  1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.dec(self.fc(z).view(z.size(0), 64, 8, 11))
        return F.interpolate(x, (self.img_h, self.img_w),
                             mode="bilinear", align_corners=False)


class VAE(nn.Module):
    def __init__(self, frame_enc, feat_dim=256, latent_dim=64,
                 img_h=244, img_w=324):
        super().__init__()
        self.encoder = VAEEncoder(frame_enc, feat_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, img_h, img_w)

    @staticmethod
    def reparameterize(mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def forward(self, x):
        mu, lv = self.encoder(x)
        z_w    = self.reparameterize(mu, lv)
        return self.decoder(z_w), mu, lv, z_w


# ── MulVPRL (GRU temporal birleştirme) ───────────────────────────────
class MulVPRL(nn.Module):
    """[B,T,1,H,W] → [B, gru_hidden]"""
    def __init__(self, frame_enc, feat_dim=256, gru_hidden=200):
        super().__init__()
        self.frame_enc = frame_enc
        self.gru = nn.GRU(feat_dim, gru_hidden, num_layers=1, batch_first=True)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.frame_enc(x.view(B*T, C, H, W)).view(B, T, -1)
        _, h  = self.gru(feats)
        return h.squeeze(0)


# ── MLP Başlıkları ────────────────────────────────────────────────────
class SteeringHead(nn.Module):
    """→ [B]  ∈ [-1, 1]"""
    def __init__(self, gru_hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Tanh(),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class CollisionHead(nn.Module):
    """→ [B]  ∈ [0, 1]"""
    def __init__(self, gru_hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ── Ana Model ─────────────────────────────────────────────────────────
class HierarchicalNavNet(nn.Module):
    """
    Üç katmanlı hiyerarşik İHA navigasyon modeli (~2.01M parametre).

    Kullanım:
        model = HierarchicalNavNet()

        # Stage I — VAE + Contrastive eğitimi
        recon, mu, lv, z_w, z_cont = model.forward_stage1(x_seq)

        # Stage II — MLP başlık eğitimi (encoder dondurulur)
        steer, collision = model.forward_stage2(x_seq)
    """
    def __init__(self, feat_dim=256, latent_dim=64,
                 gru_hidden=200, img_h=244, img_w=324):
        super().__init__()
        # Paylaşımlı encoder: VAE + MulVPRL aynı ağırlıkları kullanır
        self.shared_encoder = ResNet8Encoder(feat_dim)
        # Üst katman
        self.vae     = VAE(self.shared_encoder, feat_dim, latent_dim, img_h, img_w)
        # Orta katman
        self.mulvprl = MulVPRL(self.shared_encoder, feat_dim, gru_hidden)
        # Alt katman
        self.steering_head  = SteeringHead(gru_hidden)
        self.collision_head = CollisionHead(gru_hidden)

    # ------------------------------------------------------------------
    def forward_stage1(self, x):
        """
        Stage I ileri geçişi.
        Giriş  : x [B, SEQ_LEN, 1, H, W]
        Çıkış  :
          recon   [B, 1, H, W]    — VAE yeniden inşa
          mu      [B, latent_dim]
          lv      [B, latent_dim] — log_var
          z_w     [B, latent_dim] — waypoint gizil vektörü
          z_cont  [B, feat_dim]   — contrastive temsil (son kare)
        """
        recon, mu, lv, z_w = self.vae(x)
        z_cont = self.shared_encoder(x[:, -1])
        return recon, mu, lv, z_w, z_cont

    # ------------------------------------------------------------------
    def forward_stage2(self, x):
        """
        Stage II ileri geçişi (encoder dondurulmuş halde çalışır).
        Giriş  : x [B, SEQ_LEN, 1, H, W]
        Çıkış  :
          steer     [B]  ∈ [-1, 1]
          collision [B]  ∈ [0,  1]
        """
        h = self.mulvprl(x)
        return self.steering_head(h), self.collision_head(h)

    # ------------------------------------------------------------------
    def freeze_encoder(self):
        """Stage II için encoder ağırlıklarını dondurur."""
        for p in self.shared_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.shared_encoder.parameters():
            p.requires_grad = True

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Hızlı kontrol ────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, H, W = 4, 4, 244, 324
    x = torch.randn(B, T, 1, H, W)

    model = HierarchicalNavNet()
    total = model.count_parameters()
    print(f"Toplam eğitilebilir parametre: {total:,}  (~{total/1e6:.2f}M)")

    recon, mu, lv, z_w, z_cont = model.forward_stage1(x)
    print(f"\nStage I")
    print(f"  recon   : {recon.shape}")
    print(f"  mu      : {mu.shape}")
    print(f"  log_var : {lv.shape}")
    print(f"  z_w     : {z_w.shape}")
    print(f"  z_cont  : {z_cont.shape}")

    steer, col = model.forward_stage2(x)
    print(f"\nStage II")
    print(f"  steer     : {steer.shape}  [{steer.min():.3f}, {steer.max():.3f}]")
    print(f"  collision : {col.shape}  [{col.min():.3f}, {col.max():.3f}]")

    assert recon.shape == (B, 1, H, W)
    assert mu.shape    == (B, 64)
    assert steer.shape == (B,)
    assert col.shape   == (B,)
    print("\nTüm şekil kontrolleri geçti.")
