from src.data_preprocessing import load_and_preprocess_data
from src.model import create_model
import matplotlib.pyplot as plt

# Ruta al conjunto de datos
data_dir = 'data/raw/'

# Cargar y preprocesar los datos
train_generator, validation_generator = load_and_preprocess_data(data_dir)

# Crear el modelo
model = create_model(num_classes=train_generator.num_classes)

# Entrenar el modelo
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Guardar el modelo entrenado
model.save('models/car_classification_model.h5')

# Visualizar las métricas de entrenamiento
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
