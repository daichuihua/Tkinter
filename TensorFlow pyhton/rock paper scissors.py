import os
import tensorflow as tf

rock_dir = os.path.join('D:\\python project\\tensorflow rock_paper_scissors\\rps\\rock')
paper_dir = os.path.join('D:\\python project\\tensorflow rock_paper_scissors\\rps\\paper')
scissors_dir = os.path.join('D:\\python project\\tensorflow rock_paper_scissors\\rps\\scissors')

print('total training rock images:',len(os.listdir(rock_dir)))
print('total training paper images:',len(os.listdir(paper_dir)))
print('total training scissors images:',len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index =2

next_rock = [os.path.join(rock_dir,fname)
			    for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir,fname)
			    for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir,fname)
			    for fname in scissors_files[pic_index-2:pic_index]]

# for i,img_path in enumerate(next_rock+next_paper+next_scissors):
# 	img = mpimg.imread(img_path)
# 	plt.imshow(img)
# 	plt.axis('off')
# 	plt.show()


import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


#数据预处理
TRAINING_DIR = "D:\\python project\\tensorflow rock_paper_scissors\\rps"
training_datagen = ImageDataGenerator(
		rescale = 1./255,
		rotation_range =40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

VALIDATION_DIR = "D:\\python project\\tensorflow rock_paper_scissors\\rps-test-set"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

#构建模型
model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
		tf.keras.layers.MaxPool2D(2, 2),
		tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
		tf.keras.layers.MaxPool2D(2, 2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(512,activation='relu'),
		tf.keras.layers.Dense(3,activation='softmax')
])

model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',  #稀疏交叉熵
			  metrics=['accuracy'])


history = model.fit_generator(train_generator,
							  epochs=25,
							  validation_data=validation_generator,
							  verbose=1)

model.save("D:\\python project\\tensorflow rock_paper_scissors\\rps.h5")

import tensorflow as tf
new_model = tf.keras.models.load_model('D:\\python project\\tensorflow rock_paper_scissors\\rps.h5')
# new_model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


#创建summary
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
with summary_writer.as_default():
	for n in range(len(loss)):
		tf.summary.scalar('loss',loss[n],step=n)
		tf.summary.scalar('acc', acc[n], step=n)
	# tf.summary.image("trainpicture",train_generator[1],step=0)
# tensorboard --logdir logs


# import matplotlib.pyplot as plt
# print(history.history.keys())
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs,acc,'r',label='Training accuracy')
# plt.plot(epochs,val_acc,'b',label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()
#
# plt.plot(epochs,loss,'r',label='Training loss')
# plt.plot(epochs,val_loss,'b',label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# 使用模型
import numpy as np
# from google.colab import files
from tensorflow.keras.preprocessing import image

path ='D:\\python project\\tensorflow rock_paper_scissors\\rps-try-set\\1.jpg'
img = image.load_img(path,target_size=(150,150)) #列表
x = image.img_to_array(img) #数组
x = np.expand_dims(x,axis=0) #向量
images = np.vstack([x]) #3个通道连起来，形成一个长向量

classes = new_model.predict(images,batch_size=10)
print(classes)

# try_dir = os.path.join('D:\\python project\\tensorflow rock_paper_scissors\\rps-try-set')
# try_files = os.listdir(try_dir)
# uploaded = try_files.upload()
# for fn in uploaded.key():
# 	path = fn
# 	img = img.load_img(path,target_size=(150,150)) #列表
# 	x = image.img_to_array(img) #数组
# 	x = np.expand_dims(x,axis=0) #向量
# 	images = np.vstack([x]) #3个通道连起来，形成一个长向量
# 	classes = new_model.predict(images,batch_size=10)
# 	print(fn)
# 	print(classes)


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
#
# dataset_dir='D:\\python project\\tensorflow rock_paper_scissors\\rps-try-set'
#
# image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
# for image_filename in image_filenames:
#     print(image_filename)
