import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os 

# ==========================
# 1️⃣ Veri seti Yolları
# ==========================
train_dir = 'train/'
val_dir = 'valid/'
test_dir = 'test/'

# ==========================
# 2️⃣ Görüntü ön işleme ve Data Generator'lar
# ==========================
# Eğitim için augmentasyon
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# Validasyon ve Test için SADECE yeniden ölçekleme
test_val_datagen = ImageDataGenerator(
    rescale=1./255
)

# Veri Akışları (flow_from_directory)
# train_data.num_classes = 3 (LEVEL_1, LEVEL_2, LEVEL_3 ise) olacaktır.
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

val_data = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

test_data = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Sınıf sayısını doğrulama
NUM_CLASSES = train_data.num_classes
print(f"\n[INFO] Toplam sınıf sayısı: {NUM_CLASSES}")


# ========================================================================
# 3️⃣ AŞAMA 1: Transfer Öğrenimi (Başlangıç Fazı)
# ========================================================================

# MobileNetV2 yükle ve katmanları dondur
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# Yeni katmanlar ekle
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
# Çıktı katmanı, otomatik olarak bulunan sınıf sayısını (NUM_CLASSES) kullanır
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Modelin ilk fazı için derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Geri Çağrılar Tanımla (Faz 1 için)
checkpoint_faz1 = ModelCheckpoint(
    'best_model_faz1.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)


print("\n[INFO] Faz 1: Sadece Üst Katmanlar Eğitiliyor (5 Epoch)...")
history_phase1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    verbose=2,
    callbacks=[checkpoint_faz1]
)

# Faz 1'in en iyi ağırlıklarını yükle
model.load_weights('best_model_faz1.h5')
# Faz 2'nin başlangıç epoch değerini ayarlama
initial_epoch_for_phase2 = history_phase1.epoch[-1] + 1


# ========================================================================
# 4️⃣ AŞAMA 2: İnce Ayar (Fine-Tuning) Fazı
# ========================================================================

print("\n[INFO] Faz 2: İnce Ayar Yapılıyor (Öğrenme Oranı Düşürülüyor)...")

# 1. Base Model'i tekrar eğitilebilir yap
base_model.trainable = True

# 2. Optimizer'ı yeniden ayarla
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Geri Çağrılar Tanımla (Faz 2 için)
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=3, 
    mode='max', 
    verbose=1, 
    restore_best_weights=True
)

checkpoint_faz2 = ModelCheckpoint(
    'best_model_final.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)


# 3. Eğitime devam et
history_phase2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15, # Toplam epoch 10'dan 15'e çıkarıldı (Deneme amaçlı)
    initial_epoch=initial_epoch_for_phase2,
    verbose=2,
    callbacks=[early_stopping, checkpoint_faz2]
)

# ==========================
# 5️⃣ Modeli Kaydet
# ==========================
model.save('hair_loss_model_fine_tuned.h5')
print("\n[INFO] Modelin en iyi ağırlıkları (val_accuracy'ye göre) hair_loss_model_fine_tuned.h5 olarak kaydedildi.")


# ==========================
# 6️⃣ Doğruluk Grafiği (Birleşik)
# ==========================
history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
}

# EarlyStopping nedeniyle sadece gerçekleşen epoch'ları al
total_epochs = len(history['accuracy'])
history['accuracy'] = history['accuracy'][:total_epochs]
history['val_accuracy'] = history['val_accuracy'][:total_epochs]

plt.figure(figsize=(10, 6))
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Eğitim ve Doğrulama Doğruluğu ({NUM_CLASSES} Sınıf)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 7️⃣ Nihai Test Seti Değerlendirmesi
# ==========================
print("\n[INFO] Nihai Test Seti ile Değerlendirme Yapılıyor...")
loss, acc = model.evaluate(test_data, verbose=2)
print(f"\n=======================================================")
print(f"NIHAYI TEST SETI DOĞRULUĞU ({NUM_CLASSES} Sınıf): {acc*100:.2f}%")
print(f"=======================================================")

# ==========================
# 8️⃣ Tek bir fotoğraf tahmini fonksiyonu
# ==========================
def predict_image(img_path, model, class_indices):
    if not os.path.exists(img_path):
        print(f"HATA: '{img_path}' yolu bulunamadı.")
        return None
        
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model.predict(x, verbose=0)
    predicted_class_index = np.argmax(pred)
    
    # Sınıf etiketini bulmak için class_indices'i tersine çeviriyoruz
    labels = dict((v,k) for k,v in class_indices.items())
    predicted_label = labels[predicted_class_index]
    
    # Tahmin olasılığını da yazdıralım
    prediction_confidence = pred[0][predicted_class_index] * 100

    print(f"\nTAHMİN SONUCU: Saç dökülmesi evresi: {predicted_label}")
    print(f"Modelin Güveni: %{prediction_confidence:.2f}")
    return predicted_label

# Örnek kullanım:
# Lütfen test klasörünüzdeki (test/LEVEL_1, test/LEVEL_2 veya test/LEVEL_3) geçerli BİR resmin yolunu buraya yazın:
test_image_path = 'test/LEVEL_1/ornek_resim_1.jpg' 
# **UYARI:** Yukarıdaki yolu kendi test setinizdeki geçerli bir dosya yoluyla DEĞİŞTİRMEYİ UNUTMAYIN.
predict_image(test_image_path, model, train_data.class_indices)