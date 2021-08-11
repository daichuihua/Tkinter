import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory('C:\\Users\\Administrator\\Desktop\\Tensorflow python\\horse-or-human',
                                                    target_size=(300,300),batch_size=32,class_mode='binary')
validation_generator = validation_datagen.flow_from_directory('C:\\Users\\Administrator\\Desktop\\Tensorflow python\\validation-horse-or-human',
                                                    target_size=(300,300),batch_size=32,class_mode='binary')

from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
import tensorflow as tf

hp = HyperParameters()
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Choice('num_filters_layer0',values=[16,64],default=16), (3,3), activation='relu', input_shape=(300, 300, 3)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    for i in range(hp.Int("num_conv_layers",1,3)):
        model.add(tf.keras.layers.Conv2D(hp.Choice(f'num_filters_layer{i}',values=[16,64],default=16), (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hp.Int("hidden_units",128,512,32), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])
    return model
tuner=Hyperband(
                build_model,
                objective='val_acc',
                max_epochs=15,
                directory='horse_human_params',
                hyperparameters=hp,
                project_name='my_horse_human_project'
)
tuner.search(train_generator,epochs=10,validation_data=validation_generator)
# model.fit(train_generator,epochs=10,validation_data=validation_generator)

best_hps=tuner.get_best_hyperparameters(1)[0]
print(best_hps.values)
model = tuner.hypermodel.build(best_hps)
model.summary()