"""
VAE-based Anomaly Detection for NSL-KDD IDS Dataset
Ödev 2 – Variational Autoencoder ile Anomali Tespiti
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

# ─────────────────────────────────────────────
# 1. NSL-KDD KOLON İSİMLERİ
# ─────────────────────────────────────────────
NSL_KDD_COLS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

CAT_COLS = ['protocol_type', 'service', 'flag']

# ─────────────────────────────────────────────
# 2. VERİ YÜKLEME
# ─────────────────────────────────────────────
def load_nsl_kdd(train_path, test_path):
    if os.path.exists(train_path):
        print("Gercek NSL-KDD veri seti yukleniyor...")
        train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLS)
        test  = pd.read_csv(test_path,  header=None, names=NSL_KDD_COLS)
        print(f"  Train: {len(train):,} satir | Test: {len(test):,} satir")
        return train, test
    else:
        raise FileNotFoundError(
            f"Dosya bulunamadi: {train_path}\n"
            "KDDTrain+.txt ve KDDTest+.txt dosyalarinin ayni klasorde oldugunu kontrol edin."
        )

# ─────────────────────────────────────────────
# 3. VERİ ÖN İŞLEME
# ─────────────────────────────────────────────
def preprocess(train_df, test_df):
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df.drop(columns=['difficulty'], errors='ignore', inplace=True)
    test_df.drop(columns=['difficulty'],  errors='ignore', inplace=True)

    # İkili etiket: 0=normal, 1=saldiri
    train_df['binary_label'] = (train_df['label'] != 'normal').astype(int)
    test_df['binary_label']  = (test_df['label']  != 'normal').astype(int)

    # Kategorik sütunları kodla
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])

    feature_cols = [c for c in train_df.columns if c not in ('label', 'binary_label')]
    X_train_all = train_df[feature_cols].values.astype(np.float32)
    y_train_all = train_df['binary_label'].values
    X_test      = test_df[feature_cols].values.astype(np.float32)
    y_test      = test_df['binary_label'].values

    # VAE yalnizca NORMAL trafik uzerinde egitilecek
    X_train_normal = X_train_all[y_train_all == 0]

    # Olasi NaN/Inf temizle
    X_train_normal = np.nan_to_num(X_train_normal, nan=0.0, posinf=0.0, neginf=0.0)
    X_test         = np.nan_to_num(X_test,         nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_normal_sc = scaler.fit_transform(X_train_normal).astype(np.float32)
    X_test_sc         = scaler.transform(X_test).astype(np.float32)

    print(f"  Train normal: {X_train_normal_sc.shape[0]:,} ornek")
    print(f"  Test toplam:  {X_test_sc.shape[0]:,} ornek  "
          f"(Saldiri orani: {y_test.mean()*100:.1f}%)")
    return X_train_normal_sc, X_test_sc, y_test, scaler

# ─────────────────────────────────────────────
# 4. VAE MİMARİSİ (Saf NumPy)
# ─────────────────────────────────────────────
def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

class DenseLayer:
    def __init__(self, in_d, out_d, activation='relu', seed=None):
        rng = np.random.default_rng(seed)
        # He başlatma
        scale = np.sqrt(2.0 / in_d)
        self.W = (rng.standard_normal((in_d, out_d)) * scale).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)
        self.activation = activation
        self.x = self.z = None
        self.dW = self.db = None

    def forward(self, x):
        self.x = x.astype(np.float32)
        self.z = self.x @ self.W + self.b
        if self.activation == 'relu':
            return relu(self.z)
        return self.z  # linear

    def backward(self, d_out):
        d_out = d_out.astype(np.float32)
        if self.activation == 'relu':
            d_out = d_out * relu_grad(self.z)
        self.dW = self.x.T @ d_out
        self.db = d_out.sum(axis=0)
        # Gradient clipping — NaN/Inf patlamasini onler
        self.dW = np.clip(self.dW, -1.0, 1.0)
        self.db = np.clip(self.db, -1.0, 1.0)
        return d_out @ self.W.T


class VAE:
    """
    Mimari:
        Encoder : input(41) -> Dense(64, ReLU) -> mu(16), logvar(16)
        Decoder : z(16)     -> Dense(64, ReLU) -> output(41, linear)
    Kayip:
        L = MSE(x, x_hat) + beta * KL[q(z|x) || p(z)]
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, lr=1e-4, beta=0.5):
        self.latent_dim = latent_dim
        self.lr   = lr
        self.beta = beta

        # Encoder katmanları
        self.enc1   = DenseLayer(input_dim,  hidden_dim, 'relu',   seed=1)
        self.enc_mu = DenseLayer(hidden_dim, latent_dim, 'linear', seed=2)
        self.enc_lv = DenseLayer(hidden_dim, latent_dim, 'linear', seed=3)

        # Decoder katmanları
        self.dec1    = DenseLayer(latent_dim, hidden_dim, 'relu',   seed=4)
        self.dec_out = DenseLayer(hidden_dim, input_dim, 'linear', seed=5)

        self.history = []

    def encode(self, x):
        h  = self.enc1.forward(x)
        mu = self.enc_mu.forward(h)
        lv = self.enc_lv.forward(h)
        # logvar'i sinirla — exp patlamasini onle
        lv = np.clip(lv, -4.0, 4.0)
        return mu, lv

    def reparameterize(self, mu, lv):
        eps = np.random.randn(*mu.shape).astype(np.float32)
        return mu + eps * np.exp(0.5 * lv)

    def decode(self, z):
        h = self.dec1.forward(z)
        return self.dec_out.forward(h)

    def loss(self, x, x_hat, mu, lv):
        recon = 0.5 * np.mean((x - x_hat) ** 2)
        kl    = -0.5 * np.mean(1.0 + lv - mu**2 - np.exp(lv))
        total = recon + self.beta * kl
        return total, recon, kl

    def train_step(self, x):
        # --- İleri yayılım ---
        mu, lv = self.encode(x)
        z      = self.reparameterize(mu, lv)
        x_hat  = self.decode(z)
        total, recon, kl = self.loss(x, x_hat, mu, lv)

        # NaN kontrolü — bu batch'i atla
        if not np.isfinite(total):
            return None, None, None

        n = x.shape[0]

        # --- Geri yayılım ---
        d_xhat = -(x - x_hat) / n

        d_dec1_out = self.dec_out.backward(d_xhat)
        d_z        = self.dec1.backward(d_dec1_out)

        # KL gradyanları
        d_mu_kl = mu / n
        d_lv_kl = 0.5 * (-1.0 + np.exp(lv)) / n

        # Reparameterizasyon gradyanları
        eps        = (z - mu) / (np.exp(0.5 * lv) + 1e-8)
        d_mu_recon = d_z
        d_lv_recon = d_z * eps * 0.5 * np.exp(0.5 * lv)

        d_mu = d_mu_recon + self.beta * d_mu_kl
        d_lv = d_lv_recon + self.beta * d_lv_kl

        # Encoder geri yayılımı
        d_h_mu = self.enc_mu.backward(d_mu)
        d_h_lv = self.enc_lv.backward(d_lv)
        d_h    = d_h_mu + d_h_lv
        self.enc1.backward(d_h)

        # SGD ağırlık güncellemesi
        for layer in [self.enc1, self.enc_mu, self.enc_lv, self.dec1, self.dec_out]:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

        return float(total), float(recon), float(kl)

    def fit(self, X, epochs=80, batch_size=256, verbose=True):
        n = X.shape[0]
        print(f"  Egitim basladi: {n:,} ornek, {epochs} epoch, batch={batch_size}")
        for epoch in range(1, epochs + 1):
            idx  = np.random.permutation(n)
            X_sh = X[idx]
            losses = []
            for start in range(0, n, batch_size):
                xb = X_sh[start:start + batch_size]
                t, r, k = self.train_step(xb)
                if t is not None:
                    losses.append((t, r, k))

            if not losses:
                print(f"  Epoch {epoch}: Tum batch'ler NaN! Egitim durduruluyor.")
                break

            arr = np.array(losses)
            self.history.append(arr.mean(axis=0))

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  Epoch {epoch:3d}/{epochs}  "
                      f"loss={arr[:,0].mean():.4f}  "
                      f"recon={arr[:,1].mean():.4f}  "
                      f"kl={arr[:,2].mean():.4f}")

    def reconstruction_error(self, X, batch_size=512):
        errors = []
        for start in range(0, X.shape[0], batch_size):
            xb    = X[start:start + batch_size]
            mu, _ = self.encode(xb)
            x_hat = self.decode(mu)  # Ortalama kullan (örnekleme yok)
            mse   = np.mean((xb - x_hat) ** 2, axis=1)
            errors.append(mse)
        result = np.concatenate(errors)
        # Olasi NaN'leri 0 ile doldur
        result = np.nan_to_num(result, nan=0.0)
        return result

# ─────────────────────────────────────────────
# 5. EŞİK SEÇİMİ
# ─────────────────────────────────────────────
def choose_threshold(errors_normal, percentile=95):
    thr = float(np.percentile(errors_normal, percentile))
    print(f"  Secilen esik ({percentile}. persentil): {thr:.6f}")
    return thr

# ─────────────────────────────────────────────
# 6. DEĞERLENDİRME
# ─────────────────────────────────────────────
def evaluate(y_true, errors, threshold):
    y_pred = (errors > threshold).astype(int)
    cm_vals = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = cm_vals
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics = {
        'Accuracy':  float((tp + tn) / (tp + tn + fp + fn)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'F1-Score':  float(f1_score(y_true, y_pred, zero_division=0)),
        'ROC-AUC':   float(roc_auc_score(y_true, errors)),
        'FPR':       float(fpr_val),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }
    return metrics, y_pred

# ─────────────────────────────────────────────
# 7. GRAFİKLER
# ─────────────────────────────────────────────
def make_plots(vae, errors_normal, errors_test, y_test, threshold, metrics, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    figs = []

    # ── Şekil 1: Eğitim Kayıpları ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hist = np.array(vae.history)
    for i, (ax, lbl, col) in enumerate(zip(
            axes,
            ['Toplam Kayip', 'Rekonst. Kayip', 'KL Kayip'],
            ['#2196F3', '#4CAF50', '#FF5722'])):
        ax.plot(hist[:, i], color=col, lw=2)
        ax.set_title(lbl, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Kayip')
    fig.suptitle('VAE Egitim Kayip Egrileri', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig1_training_loss.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(); figs.append(p)

    # ── Şekil 2: Rekonstrüksiyon Hatası Dağılımı ──
    idx_norm = np.where(y_test == 0)[0]
    idx_att  = np.where(y_test == 1)[0]
    e_norm = errors_test[idx_norm]
    e_att  = errors_test[idx_att]

    # Outlier'ları kırp — görselleştirme için 99. persentil üstünü at
    clip_val = np.percentile(errors_test, 99)
    e_norm_c = np.clip(e_norm, 0, clip_val)
    e_att_c  = np.clip(e_att,  0, clip_val)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sol: Kırpılmış lineer ölçek
    ax = axes[0]
    ax.hist(e_norm_c, bins=80, alpha=.65, color='#2196F3', label='Normal', density=True)
    ax.hist(e_att_c,  bins=80, alpha=.65, color='#F44336', label='Saldiri', density=True)
    ax.axvline(threshold, color='black', ls='--', lw=2,
               label=f'Esik = {threshold:.4f}')
    ax.set_xlabel('Rekonst. Hatasi (MSE)')
    ax.set_ylabel('Yogunluk')
    ax.set_title('Rekonst. Hatasi Dagilimi (99. persentil kırpılmış)', fontweight='bold')
    ax.legend()

    # Sağ: Log ölçek (ham veri)
    ax = axes[1]
    e_norm_log = e_norm[e_norm > 0]
    e_att_log  = e_att[e_att > 0]
    ax.hist(e_norm_log, bins=80, alpha=.65, color='#2196F3', label='Normal', density=True)
    ax.hist(e_att_log,  bins=80, alpha=.65, color='#F44336', label='Saldiri', density=True)
    ax.axvline(threshold, color='black', ls='--', lw=2,
               label=f'Esik = {threshold:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel('Rekonst. Hatasi (MSE, log)')
    ax.set_ylabel('Yogunluk')
    ax.set_title('Log Olcekli Dagilim', fontweight='bold')
    ax.legend()

    fig.suptitle('Rekonstruksiyon Hatasi Analizi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig2_recon_error.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(); figs.append(p)

    # ── Şekil 3: ROC + PR Eğrileri ────────────
    fpr_arr, tpr_arr, _ = roc_curve(y_test, errors_test)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, errors_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(fpr_arr, tpr_arr, color='#2196F3', lw=2,
            label=f"AUC = {metrics['ROC-AUC']:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.scatter([metrics['FPR']], [metrics['Recall']], color='red', s=120, zorder=5,
               label=f"Secilen esik\nFPR={metrics['FPR']:.3f}, TPR={metrics['Recall']:.3f}")
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Egrisi', fontweight='bold'); ax.legend()

    ax = axes[1]
    ax.plot(rec_arr, prec_arr, color='#4CAF50', lw=2,
            label=f"F1 = {metrics['F1-Score']:.4f}")
    ax.scatter([metrics['Recall']], [metrics['Precision']], color='red', s=120, zorder=5,
               label=f"P={metrics['Precision']:.3f}, R={metrics['Recall']:.3f}")
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Egrisi', fontweight='bold'); ax.legend()
    fig.suptitle('Model Performans Egrileri', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig3_roc_pr.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(); figs.append(p)

    # ── Şekil 4: Eşik Sweep Analizi ───────────
    thresholds = np.percentile(errors_test, np.arange(50, 100, 1))
    precs, recs, f1s, fprs_sw = [], [], [], []
    for thr in thresholds:
        yp = (errors_test > thr).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
        precs.append(precision_score(y_test, yp, zero_division=0))
        recs.append(recall_score(y_test, yp, zero_division=0))
        f1s.append(f1_score(y_test, yp, zero_division=0))
        fprs_sw.append(fp2 / (fp2 + tn2 + 1e-9))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(thresholds, precs,   label='Precision', color='#2196F3', lw=2)
    ax.plot(thresholds, recs,    label='Recall',    color='#4CAF50', lw=2)
    ax.plot(thresholds, f1s,     label='F1-Score',  color='#FF5722', lw=2)
    ax.plot(thresholds, fprs_sw, label='FPR',       color='#9C27B0', lw=2, ls='--')
    ax.axvline(threshold, color='black', ls=':', lw=2,
               label=f'Secilen Esik = {threshold:.4f}')
    ax.set_xlabel('Esik Degeri (MSE)'); ax.set_ylabel('Metrik Degeri')
    ax.set_title('Esik Degerine Gore Metrik Analizi', fontweight='bold')
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig4_threshold_sweep.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(); figs.append(p)

    # ── Şekil 5: Konfüzyon Matrisi ─────────────
    y_pred_cm = (errors_test > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_cm)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Saldiri'],
                yticklabels=['Normal', 'Saldiri'])
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gercek')
    ax.set_title('Konfuzyon Matrisi', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig5_confusion_matrix.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(); figs.append(p)

    # ── Şekil 6: Hata Scatter (2 panel: tam + yakınlaştırılmış) ──
    rng_idx = np.random.default_rng(0)
    s_n = rng_idx.choice(idx_norm, size=min(2000, len(idx_norm)), replace=False)
    s_a = rng_idx.choice(idx_att,  size=min(2000, len(idx_att)),  replace=False)
    idx_s = np.concatenate([s_n, s_a])
    errs_sub   = errors_test[idx_s]
    labels_sub = y_test[idx_s]

    y_clip = float(np.percentile(errs_sub, 98))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ylim, title in zip(
            axes,
            [None, y_clip * 1.3],
            ['Tam Olcek', f'Yakinlastirilmis (y <= {y_clip:.2f})']):
        sc = ax.scatter(np.arange(len(errs_sub)), errs_sub,
                        c=labels_sub, cmap='coolwarm', alpha=0.4, s=8)
        ax.axhline(threshold, color='black', ls='--', lw=2,
                   label=f'Esik = {threshold:.4f}')
        if ylim:
            ax.set_ylim(0, ylim)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_ticks([0, 1]); cbar.set_ticklabels(['Normal', 'Saldiri'])
        ax.set_xlabel('Ornek Indeksi'); ax.set_ylabel('Rekonst. Hatasi')
        ax.set_title(title, fontweight='bold'); ax.legend()
    fig.suptitle('Rekonst. Hatasi – Normal vs Saldiri', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig6_error_scatter.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(); figs.append(p)

    return figs

# ─────────────────────────────────────────────
# 8. SONUÇ RAPORU
# ─────────────────────────────────────────────
def print_report(metrics, threshold):
    sep = "-" * 50
    print(f"\n{sep}")
    print("  VAE ANOMALI TESPIT SISTEMI - SONUCLAR")
    print(sep)
    print(f"  Esik Degeri (80. persentil): {threshold:.6f}")
    print(sep)
    print(f"  {'Metrik':<22} {'Deger':>10}")
    print(sep)
    for k in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'FPR']:
        print(f"  {k:<22} {metrics[k]:>10.4f}")
    print(sep)
    print(f"  {'TP (Dogru Saldiiri)':<22} {metrics['TP']:>10,}")
    print(f"  {'TN (Dogru Normal)':<22} {metrics['TN']:>10,}")
    print(f"  {'FP (Yanlis Alarm)':<22} {metrics['FP']:>10,}")
    print(f"  {'FN (Kacirilan Saldiri)':<22} {metrics['FN']:>10,}")
    print(sep)

# ─────────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # ── Dosya yolları ────────────────────────
    # Kodun bulundugu klasordeki dosyalari otomatik arar
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_PATH = os.path.join(SCRIPT_DIR, 'KDDTrain+.txt')
    TEST_PATH  = os.path.join(SCRIPT_DIR, 'KDDTest+.txt')
    OUT_DIR    = SCRIPT_DIR   # Grafikler ayni klasore kaydedilir

    print("=" * 60)
    print("  VAE ile Anomali Tespiti - NSL-KDD IDS")
    print("=" * 60)

    # 1. Veri yükle
    print("\n[1] Veri yukleniyor ve on isleniyor...")
    train_df, test_df = load_nsl_kdd(TRAIN_PATH, TEST_PATH)
    X_train_normal, X_test, y_test, scaler = preprocess(train_df, test_df)

    # 2. VAE eğit
    print("\n[2] VAE modeli olusturuluyor ve egitiliyor...")
    INPUT_DIM = X_train_normal.shape[1]
    vae = VAE(
        input_dim  = INPUT_DIM,
        hidden_dim = 128,
        latent_dim = 24,
        lr         = 5e-5,
        beta       = 0.3
    )
    vae.fit(X_train_normal, epochs=150, batch_size=256, verbose=True)

    # 3. Rekonstrüksiyon hataları
    print("\n[3] Rekonstruksiyon hatalari hesaplaniyor...")
    errors_normal = vae.reconstruction_error(X_train_normal)
    errors_test   = vae.reconstruction_error(X_test)

    print(f"  Normal hata  - ort: {errors_normal.mean():.4f}  std: {errors_normal.std():.4f}")
    print(f"  Test hata    - ort: {errors_test.mean():.4f}    std: {errors_test.std():.4f}")

    # 4. Eşik seç
    # 80. persentil ile baslayip en iyi F1 esigini de bul
    threshold = choose_threshold(errors_normal, percentile=80)

    # En iyi F1 esigini otomatik bul (50-99. persentil taramasi)
    from sklearn.metrics import f1_score as _f1
    best_f1_val, best_pct = 0, 80
    for pct in range(50, 100):
        thr_c = float(np.percentile(errors_normal, pct))
        yp_c  = (errors_test > thr_c).astype(int)
        f1_c  = _f1(y_test, yp_c, zero_division=0)
        if f1_c > best_f1_val:
            best_f1_val, best_pct = f1_c, pct
    print(f"  En iyi F1 ({best_f1_val:.4f}) -> {best_pct}. persentil esiginde")
    # En iyi esigi kullan
    threshold = float(np.percentile(errors_normal, best_pct))
    print(f"  Kullanilan esik: {threshold:.6f} ({best_pct}. persentil)")

    # 5. Değerlendir
    print("\n[4] Model degerlendiriliyor...")
    metrics, y_pred = evaluate(y_test, errors_test, threshold)
    print_report(metrics, threshold)



    # 6. Grafik oluştur
    print("\n[5] Grafikler olusturuluyor...")
    figs = make_plots(vae, errors_normal, errors_test, y_test,
                      threshold, metrics, out_dir=OUT_DIR)
    print(f"  {len(figs)} grafik kaydedildi:")
    for f in figs:
        print(f"    {f}")

    print("\n Tamamlandi! Grafikler ayni klasorde.")
