#加载Fashion MNIST数据集
import tensorflow as tf
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
                if(logs.get('loss')<0.1):
                        print("\nLoss is low so cancelling training!")
                        self.model.stop_training =True

callbacks = myCallback()

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_lables),(test_images,test_labels)=fashion_mnist.load_data()
print(train_images.shape)

model = keras.Sequential([
         keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
     keras.layers.MaxPooling2D(2,2),
         keras.layers.Conv2D(64,(3,3),activation='relu'),
     keras.layers.MaxPooling2D(2,2),
         keras.layers.Flatten(),
         keras.layers.Dense(128,activation=tf.nn.relu),
         keras.layers.Dense(10,activation=tf.nn.softmax)
  ])
train_images=train_images/255
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(train_images.reshape(-1,28,28,1),train_lables,epochs=2,callbacks=[callbacks])

test_images_scaled=test_images/255
model.evaluate(test_images_scaled.reshape(-1,28,28,1),test_labels)

import numpy as np
import matplotlib.pyplot as plt
# print(np.argmax(model.predict([[test_images[0]/255]])))
demo=tf.reshape(test_images[0],(-1,28,28,1))
print(np.argmax(model.predict([[demo/255]])))
print(test_labels[0])
plt.imshow(test_images[0])
plt.show()

layer_outputs =[layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input,outputs = layer_outputs)
pred = activation_model.predict(test_images[0].reshape(1,28,28,1))
plt.imshow(pred[0][0,:,:,1])
plt.show