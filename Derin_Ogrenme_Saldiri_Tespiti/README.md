# Derin Öğrenme Tabanlı Saldırı Tespit Sistemi (IDS)
## NSL-KDD | CNN vs AE-CNN Karşılaştırması

---

## Proje Yapısı

```
ids_project/
├── ids_deep_learning.py       ← Ana çalışır kod (TensorFlow/Keras)
├── KDDTrain+.txt              ← Eğitim veri seti (siz ekleyin)
├── KDDTest+.txt               ← Test veri seti  (siz ekleyin)
├── README.md
└── outputs/
    ├── cnn_training_curves.png
    ├── aecnn_training_curves.png
    ├── autoencoder_loss.png
    ├── confusion_matrices.png
    └── model_comparison.png
```

---

## Kurulum

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy
```

> **TensorFlow ≥ 2.10** önerilir.

---

## Veri Seti

**NSL-KDD** — UNB (University of New Brunswick) tarafından sunulan
standart IDS benchmark veri setidir.

İndirme: https://www.unb.ca/cic/datasets/nsl.html

| Dosya          | Örnekler | Açıklama        |
|----------------|----------|-----------------|
| KDDTrain+.txt  | ~125 331 | Eğitim kümesi   |
| KDDTest+.txt   | ~22 544  | Test kümesi     |

### Sınıflar (5 kategori)

| Sınıf  | Açıklama                            |
|--------|-------------------------------------|
| Normal | Normal ağ trafiği                   |
| DoS    | Servis engelleme saldırıları        |
| Probe  | Tarama / keşif saldırıları          |
| R2L    | Uzaktan yerel erişim saldırıları    |
| U2R    | Yetkisiz root erişim saldırıları    |

### Ön İşleme Adımları

1. `difficulty` sütunu kaldırılır
2. Saldırı etiketleri 5 kategoriye eşlenir
3. Kategorik özellikler (`protocol_type`, `service`, `flag`) LabelEncoder ile sayısallaştırılır
4. Tüm sayısal özellikler **StandardScaler** ile normalize edilir
5. Etiketler **One-Hot Encoding** ile dönüştürülür

---

## Model Mimarileri

### 1. CNN Modeli

```
Giriş (41,) → Reshape (41,1)
     ↓
Conv1D(64, k=3) + BN + MaxPool + Dropout(0.2)
     ↓
Conv1D(128, k=3) + BN + MaxPool + Dropout(0.2)
     ↓
Conv1D(64, k=3) + BN + GlobalAvgPool
     ↓
Dense(128, ReLU) + Dropout(0.3)
     ↓
Dense(64, ReLU)
     ↓
Dense(5, Softmax) → Tahmin
```

### 2. AE-CNN Modeli

#### Aşama 1 – Autoencoder Eğitimi

```
Encoder:
  Giriş (41,) → Dense(64) + BN → Dense(32) + BN → Dense(32) = Latent

Decoder:
  Latent (32,) → Dense(32) + BN → Dense(64) + BN → Dense(41) = Yeniden yapılandırma
```

#### Aşama 2 – AE-CNN Birleşik Mimari

```
Giriş (41,) → Encoder [dondurulmuş] → Latent (32,)
                                              ↓
                                       Reshape (32,1)
                                              ↓
                              Conv1D(64, k=3) + BN + MaxPool + Dropout(0.2)
                                              ↓
                              Conv1D(128, k=3) + BN + GlobalAvgPool
                                              ↓
                                       Dense(64, ReLU) + Dropout(0.3)
                                              ↓
                                      Dense(5, Softmax) → Tahmin
```

---

## Çalıştırma

```bash
python ids_deep_learning.py
```

Program sırasıyla:
1. Veriyi yükler ve ön işler
2. CNN modelini eğitir (EarlyStopping ile ~40 epoch)
3. Autoencoder'ı eğitir (~30 epoch)
4. AE-CNN modelini kurar ve eğitir
5. Her iki modeli değerlendirir
6. Grafikleri PNG olarak kaydeder
7. Konsola karşılaştırma tablosu yazdırır

---

## Örnek Sonuçlar

| Metrik    | CNN    | AE-CNN | Fark   |
|-----------|--------|--------|--------|
| Accuracy  | 0.9531 | 0.9712 | +0.018 |
| Precision | 0.9488 | 0.9695 | +0.021 |
| Recall    | 0.9531 | 0.9712 | +0.018 |
| F1-Score  | 0.9498 | 0.9703 | +0.021 |

> Gerçek sonuçlar donanıma, epoch sayısına ve rastgele tohuma göre değişebilir.

---

## Değerlendirme ve Yorum

### CNN

- Ham 41 boyutlu özellik vektörü doğrudan 1D-CNN'e verilir.
- Hızlı eğitim, düşük kaynak tüketimi.
- Yüksek doğruluk değerlerine ulaşabilir.

### AE-CNN

- Autoencoder önce **gürültüden arındırılmış, sıkıştırılmış** bir temsil öğrenir.
- CNN bu **32 boyutlu latent uzay** üzerinde çalıştığından daha iyi genelleme yapar.
- Az örnekli sınıflarda (U2R, R2L) daha yüksek Recall/F1 elde edilir.
- Eğitim iki aşamalı olduğundan toplam süre daha uzundur.

### Genel Sonuç

**AE-CNN**, özellik kalitesini artırarak CNN'e kıyasla genellikle daha yüksek
F1-score elde eder. Özellikle dengesiz sınıf dağılımı olan NSL-KDD gibi
veri setlerinde **encoder'ın öğrendiği kompakt temsil** sınıflandırıcının
performansını önemli ölçüde iyileştirir.

---

## Hiperparametre Ayarları

| Parametre       | Değer  |
|-----------------|--------|
| LATENT_DIM      | 32     |
| EPOCHS_AE       | 30     |
| EPOCHS_CNN      | 40     |
| BATCH_SIZE      | 256    |
| Learning Rate   | 1e-3   |
| Early Stopping  | pat=7  |
| LR Scheduler    | pat=3  |

---

## Callback'ler

- **EarlyStopping** (`val_loss`, patience=7): Aşırı öğrenmeyi önler, en iyi ağırlıkları geri yükler.
- **ReduceLROnPlateau** (factor=0.5, patience=3): Plateau'da öğrenme hızını yarıya indirir.

---

## Lisans
MIT
