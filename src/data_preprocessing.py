import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(data_dir, img_height=224, img_width=224, batch_size=32):
    # Crear un generador de datos de entrenamiento con aumento de datos
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # Separación para validación
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator
