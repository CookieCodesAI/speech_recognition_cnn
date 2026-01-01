import tensorflow_datasets as tfds
import tensorflow as tf
import json

def get_data():
    dataset, info = tfds.load("speech_commands", with_info=True, as_supervised=True, download=True, data_dir="./data")
    train = dataset['train']
    test = dataset['test']
    val = dataset['validation']
    return train, test, val , info

def pad_or_trim(audio, target_len=16000):
    audio = audio[:target_len]
    padding = target_len - tf.shape(audio)[0]
    audio = tf.pad(audio, [[0, padding]])
    return audio

def get_spectrogram(waveform):
    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length = 640, frame_step = 160, fft_length=640)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def preprocess(audio):
    spectrogram = get_spectrogram(audio)
    spectrogram = spectrogram[:, :129]
    spectrogram = (spectrogram - tf.reduce_mean(spectrogram)) / (
        tf.math.reduce_std(spectrogram) + 1e-6
    )
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram

def apply_preprocess(train, test, val):
    train_ds = train.map(full_preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = test.map(full_preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(128).prefetch(tf.data.AUTOTUNE)
    val_ds = val.map(full_preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(128).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds, val_ds

def get_labels(info):
    label_names = info.features['label'].names
    return label_names

def full_preprocess(audio, label):
    audio = pad_or_trim(audio)
    spec = preprocess(audio)
    return spec, label