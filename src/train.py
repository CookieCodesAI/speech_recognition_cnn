import tensorflow as tf
from preprocessing import get_data, apply_preprocess, get_labels
from model2 import make_model 
from evaluate import evaluate

#This line of code will still work even if the data is already loaded
train, test, val, info = get_data()

train_ds, test_ds, val_ds = apply_preprocess(train, test, val)

history, model = make_model(train_ds, val_ds)

#Use below code to load a presaved model for evaluation and comment out the model2 import as well as the above line 
#model = tf.keras.models.load_model("models/speech_cnn.keras")

evaluate(model, test_ds, info)