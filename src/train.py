import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from model import build_cnn
from utils import plot_history, save_history
import os

def load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

def get_data_generators(x_train, y_train, batch_size=64):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(x_train)
    return datagen.flow(x_train, y_train, batch_size=batch_size)

def train(output_path='outputs/model.h5', epochs=30, batch_size=64):
    x_train, y_train, x_test, y_test = load_and_preprocess()
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    train_gen = get_data_generators(x_tr, y_tr, batch_size=batch_size)
    steps_per_epoch = len(x_tr) // batch_size

    model = build_cnn(input_shape=x_train.shape[1:], num_classes=10)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(output_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    save_history(history, path='outputs/history.json')
    plot_history(history, outdir='outputs/figures')

    return model, history, (x_test, y_test)
