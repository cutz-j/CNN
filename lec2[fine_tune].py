import tensorflow as tf
import os

work_dir = 'd:/data'

weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'

image_height, image_width = 150, 150
train_dir = "d:/data/train"
test_dir = "d:/data/test"
no_classes = 2
no_validation = 800
epochs = 50
batch_size = 32
no_train = 2000
no_test = 800
input_shape = (image_height, image_width, 3)
epoch_steps = no_train // batch_size
test_steps = no_test // batch_size

generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
model = tf.keras.applications.DenseNet121(include_top=False)
model_tune = tf.keras.models.Sequential()
model_tune.add(tf.keras.layers.Flatten(input_shape=model.output_shape))
model_tune.add(tf.keras.layers.Dense(units=256, activation='relu'))
model_fine_tune.add(tf.keras.layers.Dropout(0.5))
model_fine_tune.add(tf.keras.layers.Dense(no_classes, activation='softmax'))

model_fine_tune.load_weights(top_model_weights_path)


tf.keras.applications.vgg16()









