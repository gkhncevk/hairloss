import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Modeli yükle
model = tf.keras.models.load_model('hair_loss_model.h5')

# Level etiketleri
labels = ["Level2", "Level3", "Level4", "Level5", "Level6", "Level7"]

# Test fotoğrafı
img_path = 'test_photo.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

pred = model.predict(x)
predicted_class = np.argmax(pred)

print("Tahmin Edilen Evre:", labels[predicted_class])
