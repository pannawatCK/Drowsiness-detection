# โหลดโมเดล Keras (.h5)
import tensorflow as tf


model = tf.keras.models.load_model('models/cnnCat2.h5')

# สร้าง TFLiteConverter และตั้งค่าตัวแปลง
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ตั้งค่า quantization (ตัวเลือก)
# ตัวอย่างนี้ใช้ quantization แบบ float16 เพื่อลดขนาดโมเดล
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# แปลงโมเดลเป็น TensorFlow Lite
tflite_model = converter.convert()

# บันทึกไฟล์ .tflite
with open('models/cnnCat2.tflite', 'wb') as f:
    f.write(tflite_model)

print("โมเดลถูกแปลงเป็น .tflite สำเร็จ!")
