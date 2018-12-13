import os
import shutil
import tensorflow as tf

work_dir = 'd:/data/'
image_names = sorted(os.listdir(os.path.join(work_dir, 'train')))

def copy_files(prefix_str, range_start, range_end, target_dir):
    ## 표본 추출해서 따로 저장 ##
    image_paths = [os.path.join(work_dir, 'train', prefix_str + '.' + str(i) + '.jpg')
    for i in range(range_start, range_end)]
    dest_dir = os.path.join(work_dir, 'data', target_dir, prefix_str)
    os.makedirs(dest_dir)
    for image_path in image_paths:
        shutil.copy(image_path, dest_dir)
#        
#copy_files('dog', 0, 1000, 'train')
#copy_files('cat', 0, 1000, 'train')
#copy_files('dog', 1000, 1400, 'test')
#copy_files('cat', 1000, 1400, 'test')

## parameter ##
image_height, image_width = 150, 150
train_dir = os.path.join(work_dir, 'data/train')
test_dir = os.path.join(work_dir, 'data/test')
num_classes = 2
num_validation = 800
epochs = 2
batch_size = 10
num_train = 2000
num_test = 800
input_shape = (image_height, image_width, 3)
epoch_steps = num_train // batch_size # 20 //--> 몫만 출력
test_steps = num_test // batch_size # 8

# image generator #
generator_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

def simple_cnn(input_shape):
    ## cnn modeling ##
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),
                                     activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model

train_images = generator_train.flow_from_directory(train_dir,
                                                   batch_size=batch_size,
                                                   target_size=(image_width, image_height))
test_images = generator_train.flow_from_directory(test_dir,
                                                   batch_size=batch_size,
                                                   target_size=(image_width, image_height))

# fit #
simple_cnn_model = simple_cnn(input_shape)
simple_cnn_model.fit_generator(
        train_images,
        steps_per_epoch=epoch_steps,
        epochs=epochs,
        validation_data=test_images,
        validation_steps=test_steps)






