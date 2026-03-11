"""
=============================================================================
Derin Öğrenme Tabanlı Saldırı Tespit Sistemi (IDS)
NSL-KDD Veri Seti ile CNN ve AE-CNN Mimarilerinin Karşılaştırması
=============================================================================
Gereksinimler:
    pip install tensorflow scikit-learn matplotlib pandas numpy seaborn

Kullanım:
    1. KDDTrain+.txt ve KDDTest+.txt dosyalarını aynı dizine koyun.
    2. python ids_deep_learning.py

Çıktılar:
    - cnn_training_curves.png       : CNN eğitim grafikleri
    - aecnn_training_curves.png     : AE-CNN eğitim grafikleri
    - autoencoder_loss.png          : Autoencoder eğitim grafiği
    - confusion_matrices.png        : Her iki modelin confusion matrix'i
    - model_comparison.png          : Karşılaştırma tablosu grafiği
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI olmayan ortamlar için backend ayarı
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────────────────────
# 1. SABİTLER VE AYARLAR
# ─────────────────────────────────────────────────────────────

TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE  = "KDDTest+.txt"
RANDOM_SEED = 42
EPOCHS_AE   = 30       # Autoencoder epoch sayısı
EPOCHS_CNN  = 40       # CNN epoch sayısı
BATCH_SIZE  = 256
LATENT_DIM  = 32       # Autoencoder gizli boyutu

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# NSL-KDD sütun isimleri (41 özellik + label + difficulty)
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Saldırı kategorileri (çok sınıflı)
ATTACK_MAP = {
    'normal': 'Normal',
    # DoS saldırıları
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS',
    'udpstorm': 'DoS', 'processtable': 'DoS', 'mailbomb': 'DoS',
    # Probe saldırıları
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L saldırıları
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
    'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L',
    'xsnoop': 'R2L', 'httptunnel': 'R2L',
    # U2R saldırıları
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
}


# ─────────────────────────────────────────────────────────────
# 2. VERİ YÜKLEME VE ÖN İŞLEME
# ─────────────────────────────────────────────────────────────

def load_nslkdd(filepath):
    """NSL-KDD txt dosyasını DataFrame olarak yükler."""
    df = pd.read_csv(filepath, header=None, names=COLUMNS)
    return df


def preprocess(df_train, df_test):
    """
    Veriyi temizler, kategorik değişkenleri encode eder,
    etiketleri haritalar ve normalize eder.

    Dönüş:
        X_train, X_test : numpy array (float32)
        y_train, y_test : numpy array (int)
        class_names     : liste
        n_classes       : int
        scaler          : eğitilmiş StandardScaler
    """
    # difficulty sütununu kaldır
    for df in [df_train, df_test]:
        df.drop(columns=['difficulty'], inplace=True, errors='ignore')

    # Saldırı etiketlerini kategorilere dönüştür
    df_train['label'] = df_train['label'].map(ATTACK_MAP).fillna('Other')
    df_test['label']  = df_test['label'].map(ATTACK_MAP).fillna('Other')

    # Kategorik özellikler: protocol_type, service, flag
    cat_cols = ['protocol_type', 'service', 'flag']

    # LabelEncoder ile tamsayıya çevir (train+test birlikte fit)
    combined = pd.concat([df_train[cat_cols], df_test[cat_cols]])
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(combined[col].astype(str))
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col]  = le.transform(df_test[col].astype(str))
        encoders[col] = le

    # Özellikler ve etiketleri ayır
    X_train = df_train.drop(columns=['label']).values.astype(np.float32)
    X_test  = df_test.drop(columns=['label']).values.astype(np.float32)

    # Etiket encoding
    le_label = LabelEncoder()
    le_label.fit(pd.concat([df_train['label'], df_test['label']]))
    y_train = le_label.transform(df_train['label'])
    y_test  = le_label.transform(df_test['label'])

    class_names = list(le_label.classes_)
    n_classes   = len(class_names)

    # Özellik normalizasyonu (StandardScaler)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, class_names, n_classes, scaler


# ─────────────────────────────────────────────────────────────
# 3. MODEL MİMARİLERİ
# ─────────────────────────────────────────────────────────────

def build_cnn_model(input_dim, n_classes):
    """
    1D-CNN tabanlı sınıflandırıcı.
    Giriş boyutu: (input_dim,)  → reshape ile (input_dim, 1) yapılır.

    Katmanlar:
        - Reshape
        - Conv1D + BatchNorm + MaxPooling (x2)
        - Flatten
        - Dense + Dropout
        - Çıkış: Softmax
    """
    inp = keras.Input(shape=(input_dim,), name='cnn_input')

    # 1D evrişim için boyut ekle: (batch, features, 1)
    x = layers.Reshape((input_dim, 1))(inp)

    # İlk CNN bloğu
    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      padding='same', name='conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # İkinci CNN bloğu
    x = layers.Conv1D(128, kernel_size=3, activation='relu',
                      padding='same', name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # Üçüncü CNN bloğu
    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      padding='same', name='conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Tam bağlantılı katmanlar
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Çıkış katmanı
    out = layers.Dense(n_classes, activation='softmax', name='cnn_output')(x)

    model = Model(inputs=inp, outputs=out, name='CNN_IDS')
    return model


def build_autoencoder(input_dim, latent_dim=32):
    """
    Tam bağlantılı Autoencoder.
    Encoder: input_dim → 64 → 32 → latent_dim
    Decoder: latent_dim → 32 → 64 → input_dim

    Dönüş:
        autoencoder : tam AE modeli (eğitim için)
        encoder     : sadece encoder kısmı (özellik çıkarımı için)
    """
    # ── Encoder ──
    enc_inp = keras.Input(shape=(input_dim,), name='ae_input')
    e = layers.Dense(64, activation='relu', name='enc_dense1')(enc_inp)
    e = layers.BatchNormalization()(e)
    e = layers.Dense(32, activation='relu', name='enc_dense2')(e)
    e = layers.BatchNormalization()(e)
    latent = layers.Dense(latent_dim, activation='relu',
                          name='latent')(e)  # Gizli temsil

    encoder = Model(inputs=enc_inp, outputs=latent, name='Encoder')

    # ── Decoder ──
    dec_inp = keras.Input(shape=(latent_dim,), name='dec_input')
    d = layers.Dense(32, activation='relu', name='dec_dense1')(dec_inp)
    d = layers.BatchNormalization()(d)
    d = layers.Dense(64, activation='relu', name='dec_dense2')(d)
    d = layers.BatchNormalization()(d)
    reconstruction = layers.Dense(input_dim, activation='linear',
                                  name='ae_output')(d)

    decoder = Model(inputs=dec_inp, outputs=reconstruction, name='Decoder')

    # ── Tam Autoencoder ──
    ae_out = decoder(encoder(enc_inp))
    autoencoder = Model(inputs=enc_inp, outputs=ae_out, name='Autoencoder')

    return autoencoder, encoder


def build_aecnn_model(encoder, latent_dim, n_classes):
    """
    AE-CNN Mimarisi:
        1. Eğitilmiş Encoder'dan özellik vektörü al (latent_dim,)
        2. Vektörü 1D CNN sınıflandırıcıya ver

    encoder katmanları dondurulur (fine-tune istenirse False yapılabilir).
    """
    # Encoder ağırlıklarını dondur
    encoder.trainable = False

    inp = keras.Input(shape=(encoder.input_shape[1],), name='aecnn_input')

    # Encoder özellik çıkarımı
    enc_out = encoder(inp)          # (batch, latent_dim)

    # CNN için boyut ekle: (batch, latent_dim, 1)
    x = layers.Reshape((latent_dim, 1))(enc_out)

    # CNN blokları
    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      padding='same', name='aecnn_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu',
                      padding='same', name='aecnn_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Tam bağlantılı katmanlar
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation='softmax',
                       name='aecnn_output')(x)

    model = Model(inputs=inp, outputs=out, name='AE_CNN_IDS')
    return model


# ─────────────────────────────────────────────────────────────
# 4. EĞİTİM YARDIMCI FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def get_callbacks(monitor='val_loss', patience=7):
    """EarlyStopping ve ReduceLROnPlateau callback'leri döndürür."""
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience,
            restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=3,
            min_lr=1e-6, verbose=0)
    ]


def evaluate_model(model, X_test, y_test, class_names, binary=False):
    """
    Modeli test seti üzerinde değerlendirir.
    binary=True → ikili sınıflandırma metrikleri
    Dönüş: metrik sözlüğü ve tahmin dizisi
    """
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    avg = 'binary' if binary else 'weighted'

    metrics = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average=avg,
                                     zero_division=0),
        'Recall':    recall_score(y_test, y_pred, average=avg,
                                  zero_division=0),
        'F1-Score':  f1_score(y_test, y_pred, average=avg,
                              zero_division=0),
    }
    return metrics, y_pred


# ─────────────────────────────────────────────────────────────
# 5. GRAFİK FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def plot_training_curves(history, title, filename):
    """Loss ve Accuracy eğitim eğrilerini çizer ve kaydeder."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Loss grafiği
    ax = axes[0]
    ax.plot(history.history['loss'], label='Train Loss',
            color='royalblue', linewidth=2)
    ax.plot(history.history['val_loss'], label='Val Loss',
            color='tomato', linewidth=2, linestyle='--')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy grafiği
    ax = axes[1]
    ax.plot(history.history['accuracy'], label='Train Acc',
            color='royalblue', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Val Acc',
            color='tomato', linewidth=2, linestyle='--')
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafik kaydedildi: {filename}")


def plot_ae_loss(history, filename):
    """Autoencoder loss eğrisini çizer."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss',
             color='royalblue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss',
             color='tomato', linewidth=2, linestyle='--')
    plt.title('Autoencoder Eğitim Kaybı', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafik kaydedildi: {filename}")


def plot_confusion_matrices(y_test, y_pred_cnn, y_pred_aecnn,
                            class_names, filename):
    """İki modelin confusion matrix'ini yan yana çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, y_pred, title in zip(
            axes,
            [y_pred_cnn, y_pred_aecnn],
            ['CNN Modeli', 'AE-CNN Modeli']):
        cm = confusion_matrix(y_test, y_pred)
        # Yüzde normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Tahmin Edilen')
        ax.set_ylabel('Gerçek')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafik kaydedildi: {filename}")


def plot_comparison_table(metrics_cnn, metrics_aecnn, filename):
    """İki modelin metriklerini tablo + çubuk grafiği olarak gösterir."""
    metric_names = list(metrics_cnn.keys())
    cnn_vals   = [metrics_cnn[m]   for m in metric_names]
    aecnn_vals = [metrics_aecnn[m] for m in metric_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('CNN vs AE-CNN Model Karşılaştırması',
                 fontsize=14, fontweight='bold')

    # ── Tablo ──
    ax_tbl = axes[0]
    ax_tbl.axis('off')
    table_data = [[f"{v:.4f}" for v in cnn_vals],
                  [f"{v:.4f}" for v in aecnn_vals]]
    table = ax_tbl.table(
        cellText=table_data,
        rowLabels=['CNN', 'AE-CNN'],
        colLabels=metric_names,
        cellLoc='center', loc='center',
        bbox=[0, 0.2, 1, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, c), cell in table.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
    ax_tbl.set_title('Performans Metrikleri Tablosu', fontweight='bold')

    # ── Çubuk grafik ──
    ax_bar = axes[1]
    x = np.arange(len(metric_names))
    width = 0.35
    bars1 = ax_bar.bar(x - width/2, cnn_vals,   width,
                       label='CNN',    color='royalblue',  alpha=0.85)
    bars2 = ax_bar.bar(x + width/2, aecnn_vals, width,
                       label='AE-CNN', color='darkorange', alpha=0.85)

    # Değer etiketleri
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_ylabel('Skor')
    ax_bar.set_title('Metrik Karşılaştırması', fontweight='bold')
    ax_bar.legend()
    ax_bar.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafik kaydedildi: {filename}")


# ─────────────────────────────────────────────────────────────
# 6. ANA AKIŞ
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Derin Öğrenme Tabanlı IDS – NSL-KDD")
    print("=" * 65)

    # ── 6.1 Veri yükleme ──────────────────────────────────────
    print("\n[1/6] Veri Yükleniyor...")
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"'{TRAIN_FILE}' bulunamadı!\n"
            "Lütfen KDDTrain+.txt ve KDDTest+.txt dosyalarını "
            "bu script ile aynı dizine koyun.\n"
            "İndirme: https://www.unb.ca/cic/datasets/nsl.html"
        )

    df_train = load_nslkdd(TRAIN_FILE)
    df_test  = load_nslkdd(TEST_FILE)
    print(f"  Eğitim: {df_train.shape}, Test: {df_test.shape}")

    # ── 6.2 Ön işleme ─────────────────────────────────────────
    print("\n[2/6] Veri Ön İşleniyor...")
    (X_train, X_test,
     y_train, y_test,
     class_names, n_classes,
     scaler) = preprocess(df_train, df_test)

    print(f"  Özellik sayısı : {X_train.shape[1]}")
    print(f"  Sınıf sayısı   : {n_classes} → {class_names}")
    print(f"  Eğitim örnekleri: {X_train.shape[0]}")
    print(f"  Test örnekleri  : {X_test.shape[0]}")

    input_dim = X_train.shape[1]

    # One-hot encoding (CNN ve AE-CNN için)
    y_train_oh = to_categorical(y_train, n_classes)
    y_test_oh  = to_categorical(y_test,  n_classes)

    # ── 6.3 CNN Modeli ─────────────────────────────────────────
    print("\n[3/6] CNN Modeli Eğitiliyor...")
    cnn_model = build_cnn_model(input_dim, n_classes)
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    cnn_model.summary()

    history_cnn = cnn_model.fit(
        X_train, y_train_oh,
        validation_split=0.1,
        epochs=EPOCHS_CNN,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1
    )

    print("  CNN eğitim grafikleri çiziliyor...")
    plot_training_curves(history_cnn,
                         'CNN Modeli – Eğitim Eğrileri',
                         'cnn_training_curves.png')

    # ── 6.4 Autoencoder Eğitimi ────────────────────────────────
    print("\n[4/6] Autoencoder Eğitiliyor...")
    autoencoder, encoder = build_autoencoder(input_dim, LATENT_DIM)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    autoencoder.summary()

    history_ae = autoencoder.fit(
        X_train, X_train,           # Giriş = Çıkış (yeniden yapılandırma)
        validation_split=0.1,
        epochs=EPOCHS_AE,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(monitor='val_loss', patience=5),
        verbose=1
    )

    print("  Autoencoder loss grafiği çiziliyor...")
    plot_ae_loss(history_ae, 'autoencoder_loss.png')

    # ── 6.5 AE-CNN Modeli ──────────────────────────────────────
    print("\n[5/6] AE-CNN Modeli Oluşturuluyor ve Eğitiliyor...")
    aecnn_model = build_aecnn_model(encoder, LATENT_DIM, n_classes)
    aecnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    aecnn_model.summary()

    history_aecnn = aecnn_model.fit(
        X_train, y_train_oh,
        validation_split=0.1,
        epochs=EPOCHS_CNN,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1
    )

    print("  AE-CNN eğitim grafikleri çiziliyor...")
    plot_training_curves(history_aecnn,
                         'AE-CNN Modeli – Eğitim Eğrileri',
                         'aecnn_training_curves.png')

    # ── 6.6 Değerlendirme ─────────────────────────────────────
    print("\n[6/6] Modeller Değerlendiriliyor...")

    metrics_cnn,   y_pred_cnn   = evaluate_model(
        cnn_model,   X_test, y_test, class_names)
    metrics_aecnn, y_pred_aecnn = evaluate_model(
        aecnn_model, X_test, y_test, class_names)

    # ── Confusion Matrix ──
    plot_confusion_matrices(y_test, y_pred_cnn, y_pred_aecnn,
                            class_names, 'confusion_matrices.png')

    # ── Karşılaştırma tablosu ──
    plot_comparison_table(metrics_cnn, metrics_aecnn,
                          'model_comparison.png')

    # ── Sonuç raporu ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SONUÇ RAPORU")
    print("=" * 65)

    header = f"{'Metrik':<15} {'CNN':>12} {'AE-CNN':>12} {'Fark':>12}"
    print(header)
    print("-" * len(header))

    for metric in metrics_cnn:
        v_cnn   = metrics_cnn[metric]
        v_aecnn = metrics_aecnn[metric]
        diff    = v_aecnn - v_cnn
        sign    = '+' if diff >= 0 else ''
        print(f"{metric:<15} {v_cnn:>12.4f} {v_aecnn:>12.4f} "
              f"{sign}{diff:>11.4f}")

    print("\nDetaylı Sınıflandırma Raporu (CNN):")
    print(classification_report(y_test, y_pred_cnn,
                                 target_names=class_names, zero_division=0))

    print("Detaylı Sınıflandırma Raporu (AE-CNN):")
    print(classification_report(y_test, y_pred_aecnn,
                                 target_names=class_names, zero_division=0))

    # ── Yorum ──
    print("\n" + "=" * 65)
    print("  DEĞERLENDİRME VE YORUM")
    print("=" * 65)
    best_model = "AE-CNN" if metrics_aecnn['F1-Score'] >= metrics_cnn['F1-Score'] else "CNN"
    f1_diff = abs(metrics_aecnn['F1-Score'] - metrics_cnn['F1-Score'])

    print(f"""
  CNN Modeli:
    • Ham özellikler doğrudan 1D-CNN ile işlenir.
    • Daha hızlı eğitim, daha düşük hesaplama maliyeti.
    • NSL-KDD gibi tablolar verisi için iyi temel performans verir.

  AE-CNN Modeli:
    • Autoencoder önce gürültüden arındırılmış, sıkıştırılmış
      özellik temsili öğrenir ({LATENT_DIM} boyutlu latent uzay).
    • CNN bu zengin temsil üzerinde sınıflandırma yapar.
    • Transfer öğrenme prensibi: Encoder dondurulup CNN'e rehberlik eder.
    • Az örnekli sınıflarda (U2R, R2L) daha iyi genelleme sağlayabilir.

  ► En iyi model (F1-Score): {best_model}  (fark: {f1_diff:.4f})
""")

    print("Tüm grafikler mevcut dizine kaydedildi.")
    print("=" * 65)


if __name__ == "__main__":
    main()
