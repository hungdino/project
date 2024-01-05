from model import MyModel
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the model from the checkpoint
model = load_model('training/model-14-0.95.h5')

# Adjust the learning rate
new_learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate=new_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

additional_epochs = 3
data_dir = 'data/train'
train_dir = 'data/train_split'
val_dir = 'data/val_split'

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
# Continue training with the new settings
history = model.fit(
    train_generator,
    epochs=additional_epochs,  # Set the number of additional epochs you want to train for
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.n // val_generator.batch_size
)
model.save('trained_modelEpo15-17.h5')
