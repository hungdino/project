from model import MyModel
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import shutil


data_dir = 'data/train'
train_dir = 'data/train_split'
val_dir = 'data/val_split'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    train_class_path = os.path.join(train_dir, class_dir)
    val_class_path = os.path.join(val_dir, class_dir)
    
    # Create class directories in train and validation folders
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(val_class_path, exist_ok=True)

    # Get all images in class directory
    images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2) # 20% for validation

    # Copy images to respective directories
    for img in train_imgs:
        shutil.copy(img, train_class_path)
    for img in val_imgs:
        shutil.copy(img, val_class_path)


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True) 

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(224, 224), 
                                                    batch_size=256, 
                                                    class_mode='categorical') 

val_generator = val_datagen.flow_from_directory(val_dir, 
                                                target_size=(224, 224),  
                                                batch_size=256,  
                                                class_mode='categorical')  

model = MyModel()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

checkpoint = ModelCheckpoint(
    'training/model-{epoch:02d}-{val_accuracy:.2f}.h5', 
    monitor='val_accuracy', 
    verbose=1,
    save_best_only=True,
    mode='max', 
    save_weights_only=False
)

history = model.fit(
    train_generator,
    epochs = 30,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.n // val_generator.batch_size,
    callbacks=[checkpoint]
)

model.save('trained_model_base.h5')