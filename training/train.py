import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
            '--input_path',
            help="Dataset path",
            required=True)

    parser.add_argument('-model',
            '--model_path',
            help="Output path for models",
            required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    df = df.drop('index',axis=1).drop('bankrot',axis=1)
    X_train = df.drop('target',axis=1).values
    Y_train = df['target'].values
    X_train, Y_train = shuffle(X_train, Y_train)
    Y_train = to_categorical(Y_train)

    # Model build
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train[0].shape)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.003)),
        keras.layers.Dense(86, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.002)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.tanh),
        keras.layers.Dense(16, activation=tf.nn.relu, bias_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # Callback functions
    tensorboard = keras.callbacks.TensorBoard(log_dir='./train_log')
    # Run training
    model.fit(X_train, Y_train, epochs=100, callbacks=[tensorboard], validation_split=0.2)
    # Save model
    model.save(args.model_path)
if __name__ == "__main__":
    main()
