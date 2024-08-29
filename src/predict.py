import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Cargar el modelo entrenado
model = load_model('models/car_classification_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Crear un batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Obtener la clase con la mayor probabilidad
    class_idx = np.argmax(score)
    class_name = list(train_generator.class_indices.keys())[class_idx]
    confidence = 100 * np.max(score)

    print(f"Predicción: {class_name} con una confianza de {confidence:.2f}%")

# Ejemplo de predicción
predict_image('ruta/a/imagen/nueva.jpg')
