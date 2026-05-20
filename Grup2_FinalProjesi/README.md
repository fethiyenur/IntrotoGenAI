# VAE Tabanlı Hiyerarşik İHA Navigasyonu — Uygulama Kılavuzu

## Dosya Yapısı

```
proje/
├── dataset.py        # Aşama 1 — HDIN veri yükleyici
├── model.py          # Aşama 2 — Hiyerarşik sinir ağı mimarisi
├── train.py          # Aşama 3 — İki aşamalı eğitim scripti
├── requirements.txt
├── README.md
└── HDIN/             # ← Veri setini buraya koyun
    ├── training/
    │   ├── steer001/
    │   │   ├── images/
    │   │   │   ├── 00000.png
    │   │   │   ├── 00001.png
    │   │   │   └── ...
    │   │   └── labels.txt       # iki sütun: timestamp, value
    │   ├── collision001/
    │   │   ├── images/
    │   │   └── labels.txt       # tek sütun: 0 ya da 1
    │   └── ...
    ├── validation/    (aynı yapı)
    └── testing/       (aynı yapı)
```

---

## Kurulum

```bash
pip install torch torchvision pillow numpy
```

GPU için (CUDA 12.x):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Çalıştırma

### 1. Hızlı Test (fast_test) — Kurulumun çalıştığını doğrula
```bash
python train.py --data_root HDIN --fast_test
```
> %5 veri, 3+2 epoch, batch=8. Yaklaşık 5-10 dakika (CPU).

---

### 2. Tam Eğitim (önerilen)
```bash
python train.py \
  --data_root      HDIN \
  --stage1_epochs  50   \
  --stage2_epochs  30   \
  --batch_size     48   \
  --lr_stage1      1e-3 \
  --lr_stage2      5e-4 \
  --beta           1.0  \
  --lam            0.5  \
  --num_workers    4    \
  --ckpt_dir       checkpoints
```

---

### 3. Sadece Stage II (Stage I checkpoint'i var)
```bash
python train.py \
  --data_root    HDIN \
  --skip_stage1       \
  --resume       checkpoints/stage1_best.pt \
  --stage2_epochs 30
```

---

### 4. Veri yükleyiciyi ayrıca test et
```bash
python dataset.py HDIN/training
```

### 5. Model boyutunu kontrol et
```bash
python model.py
# Çıktı: Toplam eğitilebilir parametre: 2,012,xxx (~2.01M)
```

---

## Argümanlar

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--data_root` | `HDIN` | HDIN ana dizini |
| `--stage1_epochs` | `50` | Stage I epoch sayısı |
| `--stage2_epochs` | `30` | Stage II epoch sayısı |
| `--batch_size` | `8` | Batch boyutu |
| `--lr_stage1` | `1e-3` | Stage I öğrenme hızı |
| `--lr_stage2` | `5e-4` | Stage II öğrenme hızı |
| `--beta` | `1.0` | KL kayıp ağırlığı (β) |
| `--lam` | `0.5` | InfoNCE kayıp ağırlığı (λ) |
| `--num_workers` | `4` | DataLoader worker sayısı |
| `--ckpt_dir` | `checkpoints` | Checkpoint kayıt dizini |
| `--fast_test` | `False` | Hızlı test modu |
| `--skip_stage1` | `False` | Stage I'i atla |
| `--resume` | `None` | Stage I checkpoint yolu |

---

## Kayıp Fonksiyonları

**Stage I:**
```
L_I = L_recon + β·L_KL + λ·L_InfoNCE
```
- `L_recon`: VAE yeniden inşa kaybı (MSE)
- `L_KL`: KL ıraksaması
- `L_InfoNCE`: Contrastive temsil kaybı

**Stage II:**
```
L_II = L_MSE(steering) + L_BCE(collision)
```

---

## Checkpoint Dosyaları

```
checkpoints/
├── stage1_best.pt    # Stage I — en iyi validation kaybı
├── stage2_best.pt    # Stage II — en iyi steering RMSE
└── final_model.pt    # Son model (test sonuçlarıyla)
```

---

## Beklenen Hedef Metrikler (tam eğitim)

| Metrik | Ara Sonuç (fast_test) | Hedef |
|---|---|---|
| Stage I Loss | ~1.68 | < 0.80 |
| Steering RMSE | ~0.597 | < 0.25 |
| Collision Acc | ~%65 | > %85 |
