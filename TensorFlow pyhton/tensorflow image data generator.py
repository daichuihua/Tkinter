from tensorflow.keras.preprocessing.image import ImageDataGenerator

#创建两个数据生成器，指定scaling范围0~1
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

#指定训练数据文件夹
train_generator = train_datagen.flow_from_directory(
        'D:/python project/tensorflow image data generator/horse-or-human',#训练数据所在文件夹
        target_size=(300,300),#指定输出尺寸
        batch_size=32,
        class_mode='binary') #指定二分类
validation_generator = validation_datagen.flow_from_directory(
        'D:/python project/tensorflow image data generator/validation-horse-or-human',
        target_size=(300,300),
        batch_size=32,
        class_mode='binary')