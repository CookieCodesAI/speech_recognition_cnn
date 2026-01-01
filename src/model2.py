import tensorflow as tf
from tensorflow.keras import layers, models

def make_model(train_ds, val_ds):
    num_labels = 12
    data_augmentation = tf.keras.Sequential([
        layers.RandomTranslation(0.1,0.0)
    ])
    model = models.Sequential([
        layers.Input(shape=(98, 129, 1)),
        data_augmentation,

        layers.Conv2D(32, 3, padding = "same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding = "same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        
        layers.GlobalAveragePooling2D(), 

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels)
    ])
    model.compile(optimizer = "adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(train_ds, validation_data = val_ds, epochs=20)
    model.save("models/speech_cnn.keras")
    return history, model
    #accuracy: ~91%