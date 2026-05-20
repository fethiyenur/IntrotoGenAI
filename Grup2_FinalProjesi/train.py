"""
train.py – İki Aşamalı Eğitim Scripti
========================================
Stage I : VAE (recon + beta*KL) + InfoNCE (contrastive) kayıpları
Stage II : Encoder dondurulur; sadece MLP başlıkları MSE + BCE ile eğitilir

Kullanım:
    # Tam eğitim
    python train.py --data_root HDIN --stage1_epochs 50 --stage2_epochs 30

    # Hızlı test (%5 veri, az epoch)
    python train.py --data_root HDIN --fast_test

    # Sadece Stage II (önceden kaydedilmiş checkpoint ile)
    python train.py --data_root HDIN --skip_stage1 --resume checkpoints/stage1_best.pt
"""

import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from dataset import build_dataloader, HDINDataset, collate_multitask
from model   import HierarchicalNavNet

# ── Kayıp fonksiyonları ───────────────────────────────────────────────

def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, dict]:
    """L_recon (MSE) + beta * L_KL"""
    l_recon = F.mse_loss(recon, target)
    l_kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total   = l_recon + beta * l_kl
    return total, {"recon": l_recon.item(), "kl": l_kl.item()}


def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """
    Basitleştirilmiş InfoNCE: aynı batch'teki iki farklı
    augmentation görünümü arasında contrastive kayıp.
    z_a, z_b: [B, feat_dim]  (L2 normalize edilmeden önce)
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    B   = z_a.size(0)
    # [2B, feat_dim]
    z   = torch.cat([z_a, z_b], dim=0)
    sim = torch.mm(z, z.t()) / temperature          # [2B, 2B]
    # Köşegen maskesi (kendisiyle benzerlik)
    mask = torch.eye(2 * B, device=z.device).bool()
    sim  = sim.masked_fill(mask, -1e9)
    # Pozitif çiftler: (i, i+B) ve (i+B, i)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])         # [2B]
    return F.cross_entropy(sim, labels)


def stage1_loss(recon, target, mu, log_var, z_cont_a, z_cont_b,
                beta=1.0, lam=0.5, temperature=0.07):
    """L_I = L_VAE + lambda * L_InfoNCE"""
    l_vae, vae_parts = vae_loss(recon, target, mu, log_var, beta)
    l_nce = info_nce_loss(z_cont_a, z_cont_b, temperature)
    total = l_vae + lam * l_nce
    return total, {**vae_parts, "nce": l_nce.item(), "total": total.item()}


def stage2_loss(steer_pred, steer_gt, col_pred, col_gt):
    """L_II = L_MSE(steering) + L_BCE(collision)"""
    losses = {}
    total  = torch.tensor(0.0, device=steer_pred.device
                          if steer_pred is not None else col_pred.device)

    if steer_pred is not None and steer_gt is not None:
        l_mse       = F.mse_loss(steer_pred, steer_gt)
        losses["mse"] = l_mse.item()
        total       = total + l_mse

    if col_pred is not None and col_gt is not None:
        l_bce         = F.binary_cross_entropy(col_pred, col_gt)
        losses["bce"] = l_bce.item()
        total         = total + l_bce

    losses["total"] = total.item()
    return total, losses

# ── Yardımcı: veri alt kümesi ─────────────────────────────────────────

def subset_loader(root_dir: str, fraction: float, batch_size: int,
                  augment: bool, shuffle: bool, num_workers: int):
    """Hızlı test için veri setinin fraction kadarını kullan."""
    from torch.utils.data import DataLoader
    ds   = HDINDataset(root_dir=root_dir, augment=augment)
    n    = max(1, int(len(ds) * fraction))
    idx  = list(range(n))
    sub  = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_multitask,
                      pin_memory=torch.cuda.is_available(), drop_last=False)

# ── Metrikleri hesapla ────────────────────────────────────────────────

@torch.no_grad()
def evaluate_stage2(model, loader, device):
    model.eval()
    steer_sq, steer_n = 0.0, 0
    col_correct, col_n = 0, 0

    for batch in loader:
        # Steering
        if batch["steer_images"] is not None:
            imgs = batch["steer_images"].to(device)
            lbl  = batch["steer_labels"].to(device)
            pred, _ = model.forward_stage2(imgs)
            steer_sq += ((pred - lbl) ** 2).sum().item()
            steer_n  += lbl.size(0)

        # Collision
        if batch["collision_images"] is not None:
            imgs = batch["collision_images"].to(device)
            lbl  = batch["collision_labels"].to(device)
            _, pred = model.forward_stage2(imgs)
            col_correct += ((pred > 0.5).float() == lbl).sum().item()
            col_n       += lbl.size(0)

    rmse = math.sqrt(steer_sq / steer_n) if steer_n > 0 else float("nan")
    acc  = col_correct / col_n           if col_n  > 0 else float("nan")
    return rmse, acc

# ── Stage I eğitimi ───────────────────────────────────────────────────

def train_stage1(model, train_loader, val_loader, args, device, ckpt_dir):
    """
    Optimizer kapsamı: shared_encoder + VAE (decoder dahil)
    MLP başlıkları Stage I'de optimize edilmez.
    """
    params_s1 = (
        list(model.shared_encoder.parameters()) +
        list(model.vae.encoder.fc_mu.parameters()) +
        list(model.vae.encoder.fc_logvar.parameters()) +
        list(model.vae.decoder.parameters())
    )
    optimizer = torch.optim.Adam(params_s1, lr=args.lr_stage1,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage1_epochs, eta_min=1e-6)

    best_loss = float("inf")
    print(f"\n{'='*60}")
    print(f"  STAGE I — {args.stage1_epochs} epoch  |  beta={args.beta}  lam={args.lam}")
    print(f"{'='*60}")

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        running = {"recon": 0, "kl": 0, "nce": 0, "total": 0}
        n_batch = 0

        for batch in train_loader:
            # Stage I için hem steer hem collision görüntülerini kullan
            imgs_list = []
            if batch["steer_images"] is not None:
                imgs_list.append(batch["steer_images"].to(device))
            if batch["collision_images"] is not None:
                imgs_list.append(batch["collision_images"].to(device))
            if not imgs_list:
                continue
            x = torch.cat(imgs_list, dim=0)   # [B_total, T, 1, H, W]

            # Hedef: dizinin son karesi
            target = x[:, -1]                  # [B, 1, H, W]

            # İki farklı "view" üret (ilk T-1 kare vs son T-1 kare)
            recon, mu, lv, z_w, z_cont_a = model.forward_stage1(x)
            # İkinci view: son kare yerine ilk kareyi al
            z_cont_b = model.shared_encoder(x[:, 0])

            loss, parts = stage1_loss(
                recon, target, mu, lv, z_cont_a, z_cont_b,
                beta=args.beta, lam=args.lam,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params_s1, max_norm=5.0)
            optimizer.step()

            for k in running:
                running[k] += parts.get(k, 0)
            n_batch += 1

        scheduler.step()

        avg = {k: v / max(n_batch, 1) for k, v in running.items()}
        print(f"[S1] Epoch {epoch:3d}/{args.stage1_epochs} | "
              f"total={avg['total']:.4f}  recon={avg['recon']:.4f}  "
              f"kl={avg['kl']:.4f}  nce={avg['nce']:.4f}")

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            path = ckpt_dir / "stage1_best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "loss": best_loss}, path)
            print(f"           ✓ Checkpoint kaydedildi → {path}")

    print(f"  Stage I tamamlandı.  En iyi kayıp: {best_loss:.4f}\n")

# ── Stage II eğitimi ──────────────────────────────────────────────────

def train_stage2(model, train_loader, val_loader, args, device, ckpt_dir):
    """
    Encoder dondurulur.
    Optimizer kapsamı: SADECE steering_head + collision_head parametreleri.
    Bu sayede parametre kapsam dışı kalma hatası ortadan kalkar.
    """
    model.freeze_encoder()

    # Yalnızca MLP başlık parametrelerini optimizer'a ver
    mlp_params = (
        list(model.steering_head.parameters()) +
        list(model.collision_head.parameters())
    )
    optimizer = torch.optim.Adam(mlp_params, lr=args.lr_stage2,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                 gamma=0.5)

    best_rmse = float("inf")
    print(f"\n{'='*60}")
    print(f"  STAGE II — {args.stage2_epochs} epoch  (encoder donduruldu)")
    print(f"{'='*60}")

    for epoch in range(1, args.stage2_epochs + 1):
        model.train()
        running = {"mse": 0, "bce": 0, "total": 0}
        n_batch = 0

        for batch in train_loader:
            steer_imgs = batch["steer_images"]
            steer_lbl  = batch["steer_labels"]
            col_imgs   = batch["collision_images"]
            col_lbl    = batch["collision_labels"]

            # Hem steering hem collision yoksa atla
            if steer_imgs is None and col_imgs is None:
                continue

            # Steering kolu
            s_pred, c_pred_from_steer = None, None
            if steer_imgs is not None:
                si   = steer_imgs.to(device)
                sl   = steer_lbl.to(device)
                s_p, _ = model.forward_stage2(si)
                s_pred, s_gt = s_p, sl
            else:
                s_pred, s_gt = None, None

            # Collision kolu
            if col_imgs is not None:
                ci   = col_imgs.to(device)
                cl   = col_lbl.to(device)
                _, c_p = model.forward_stage2(ci)
                c_pred, c_gt = c_p, cl
            else:
                c_pred, c_gt = None, None

            loss, parts = stage2_loss(s_pred, s_gt, c_pred, c_gt)

            if loss.item() == 0.0:
                continue   # boş batch

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mlp_params, max_norm=5.0)
            optimizer.step()

            for k in ("mse", "bce", "total"):
                running[k] += parts.get(k, 0)
            n_batch += 1

        scheduler.step()
        avg = {k: v / max(n_batch, 1) for k, v in running.items()}

        # Doğrulama metrikleri
        rmse, acc = evaluate_stage2(model, val_loader, device)
        print(f"[S2] Epoch {epoch:3d}/{args.stage2_epochs} | "
              f"loss={avg['total']:.4f}  mse={avg['mse']:.4f}  "
              f"bce={avg['bce']:.4f} || "
              f"val_RMSE={rmse:.4f}  val_Acc={acc:.2%}")

        if not math.isnan(rmse) and rmse < best_rmse:
            best_rmse = rmse
            path = ckpt_dir / "stage2_best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "rmse": rmse, "acc": acc}, path)
            print(f"           ✓ Checkpoint kaydedildi → {path}")

    model.unfreeze_encoder()
    print(f"  Stage II tamamlandı.  En iyi RMSE: {best_rmse:.4f}\n")

# ── Argümanlar ────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="HDIN Hiyerarşik Navigasyon Eğitimi")
    p.add_argument("--data_root",      default="HDIN",       help="HDIN ana dizini")
    p.add_argument("--stage1_epochs",  type=int, default=50)
    p.add_argument("--stage2_epochs",  type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--lr_stage1",      type=float, default=1e-3)
    p.add_argument("--lr_stage2",      type=float, default=5e-4)
    p.add_argument("--beta",           type=float, default=1.0,  help="KL ağırlığı")
    p.add_argument("--lam",            type=float, default=0.5,  help="InfoNCE ağırlığı")
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--ckpt_dir",       default="checkpoints")
    p.add_argument("--fast_test",      action="store_true",
                   help="%5 veri, 3+2 epoch, batch=8 ile hızlı test")
    p.add_argument("--skip_stage1",    action="store_true",
                   help="Stage I'i atla (resume ile kullan)")
    p.add_argument("--resume",         default=None,
                   help="Stage I checkpoint yolu (--skip_stage1 ile)")
    return p.parse_args()

# ── Ana giriş noktası ─────────────────────────────────────────────────

def main():
    args = get_args()

    # fast_test modu
    if args.fast_test:
        args.stage1_epochs = 3
        args.stage2_epochs = 2
        args.batch_size    = 8
        print("⚡ fast_test modu: %5 veri, 3+2 epoch, batch=8")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cihaz : {device}")
    print(f"Veri  : {args.data_root}")

    # ── DataLoader'lar ──────────────────────────────────────────────
    fraction    = 0.05 if args.fast_test else 1.0
    nw          = 0    if args.fast_test else args.num_workers

    def _loader(split, augment, shuffle):
        root = os.path.join(args.data_root, split)
        if fraction < 1.0:
            return subset_loader(root, fraction, args.batch_size,
                                 augment, shuffle, nw)
        return build_dataloader(root, args.batch_size, augment,
                                shuffle, nw)

    train_loader = _loader("training",   augment=True,  shuffle=True)
    val_loader   = _loader("validation", augment=False, shuffle=False)

    print(f"Train örnekleri : {len(train_loader.dataset)}")
    print(f"Val   örnekleri : {len(val_loader.dataset)}")

    # ── Model ──────────────────────────────────────────────────────
    model = HierarchicalNavNet().to(device)
    print(f"Parametre sayısı: {model.count_parameters():,}")

    # ── Stage I ────────────────────────────────────────────────────
    if args.skip_stage1:
        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Stage I checkpoint yüklendi: {args.resume}")
        else:
            print("Uyarı: --skip_stage1 verildi ama --resume yok; "
                  "rastgele ağırlıklarla Stage II başlıyor.")
    else:
        train_stage1(model, train_loader, val_loader, args, device, ckpt_dir)
        # En iyi Stage I ağırlıklarını yükle
        best_s1 = ckpt_dir / "stage1_best.pt"
        if best_s1.exists():
            ckpt = torch.load(best_s1, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Stage I en iyi ağırlıkları yüklendi.")

    # ── Stage II ───────────────────────────────────────────────────
    train_stage2(model, train_loader, val_loader, args, device, ckpt_dir)

    # ── Son değerlendirme ──────────────────────────────────────────
    test_loader = _loader("testing", augment=False, shuffle=False)
    rmse, acc   = evaluate_stage2(model, test_loader, device)
    print(f"\n{'='*60}")
    print(f"  TEST SONUÇLARI")
    print(f"  Steering RMSE : {rmse:.4f}")
    print(f"  Collision Acc : {acc:.2%}")
    print(f"{'='*60}\n")

    # Son model kaydı
    final_path = ckpt_dir / "final_model.pt"
    torch.save({"model": model.state_dict(),
                "test_rmse": rmse, "test_acc": acc}, final_path)
    print(f"Final model kaydedildi → {final_path}")


if __name__ == "__main__":
    main()
