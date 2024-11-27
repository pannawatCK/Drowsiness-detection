# import tensorflow as tf
# import numpy as np

# # โหลดโมเดล TFLite
# interpreter = tf.lite.Interpreter(model_path='models/cnnCat2.tflite')
# interpreter.allocate_tensors()

# # เตรียมข้อมูลตัวอย่าง (ใช้ภาพจาก data/test)
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # ตัวอย่างอินพุต (สมมติว่าภาพ grayscale ขนาด 24x24)
# test_image = np.random.rand(1, 24, 24, 1).astype(np.float32)

# # ใส่ข้อมูลเข้าไปในโมเดล
# interpreter.set_tensor(input_details[0]['index'], test_image)

# # รันโมเดล
# interpreter.invoke()

# # ดึงผลลัพธ์
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print("ผลลัพธ์การพยากรณ์:", output_data)
