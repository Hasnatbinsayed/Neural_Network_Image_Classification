import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import visualize_predictions

def load_testset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    return x_test, y_test

def run_predictions(model_path='outputs/model.h5'):
    model = load_model(model_path)
    x_test, y_test = load_testset()
    visualize_predictions(model, x_test, y_test, num=12)

if __name__ == '__main__':
    run_predictions()
