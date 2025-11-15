import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os # Dosya yolunu kontrol etmek için ekledik

# ====================================================================
# 1. MODAL YÜKLEME (Eğitim sonrası kaydettiğiniz dosya adını kullanın)
# ====================================================================
MODEL_PATH = 'hair_loss_model_fine_tuned.h5' 
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model başarıyla yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"❌ HATA: Model yüklenemedi. '{MODEL_PATH}' dosyasının varlığını kontrol edin.")
    print(f"Hata detayı: {e}")
    exit()

# ====================================================================
# 2. ETİKETLERİ DÜZELTME (3 sınıfa göre ayarlayın)
# ====================================================================
# Bu etiketler, veri setinizdeki klasör isimleriyle (LEVEL_1, LEVEL_2, LEVEL_3) eşleşmelidir.
labels = ["LEVEL_1", "LEVEL_2", "LEVEL_3"] 
INPUT_SIZE = (224, 224) # Eğitilen modelin beklediği boyut

# ====================================================================
# 3. GÖRSELİ YÜKLEME VE ÖN İŞLEME
# ====================================================================
TEST_IMAGE_PATH = 'test_photo.jpg' 

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"\n❌ HATA: Test görseli bulunamadı: '{TEST_IMAGE_PATH}'")
    print("Lütfen dosya yolunu kontrol edin veya resmi betiğin yanına taşıyın.")
else:
    # Görseli yükle ve modelin beklediği formata dönüştür
    img = image.load_img(TEST_IMAGE_PATH, target_size=INPUT_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) # (1, 224, 224, 3) şekline getir
    x = x / 255.0 # Normalizasyon

    # ====================================================================
    # 4. TAHMİN YAPMA
    # ====================================================================
    pred = model.predict(x, verbose=0)
    
    # En yüksek olasılığa sahip indeksi bul
    predicted_class_index = np.argmax(pred)
    
    # İndeksi etikete çevir
    predicted_label = labels[predicted_class_index]
    
    # Güven seviyesini hesapla
    confidence = pred[0][predicted_class_index] * 100

    # ====================================================================
    # 5. SONUÇLARI GÖSTERME
    # ====================================================================
    print("\n==============================================")
    print(f"TAHMİN EDİLEN EVRE: {predicted_label}")
    print(f"Modelin Güveni: %{confidence:.2f}")
    print("==============================================")