#Original CNN architecture used, however it did not have great accuracy. 
#Switched to model2.py

import tensorflow as tf
from tensorflow.keras import layers, models

def make_model(train_ds, val_ds):
    num_labels = 12
    model = models.Sequential([
        layers.Input(shape=(98, 129, 1)),

        layers.Conv2D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        layers.GlobalAveragePooling2D(), 

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels)
    ])
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(train_ds, validation_data = val_ds, epochs=20)
    model.save("models/speech_cnn.keras")
    return history, model

    #accuracy: ~63%