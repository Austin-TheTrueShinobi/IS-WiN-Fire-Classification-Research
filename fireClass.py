import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import shutil
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameter

# Define directories
base_dir = "C:\\Users\\mvppr\\Desktop\\comms\\SpecialTopicResearch\\FLAME3\\Version 2\\Shoetank [DRI]\\Images"
fire_dir = os.path.join(base_dir, "Fire")
no_fire_dir = os.path.join(base_dir, "No Fire")

# Create directories for train, validation, and test
train_dir = os.path.join(base_dir, "Train")
validation_dir = os.path.join(base_dir, "Validation")
test_dir = os.path.join(base_dir, "Test")

# Create subdirectories for classes in train, validation, and test
train_fire_dir = os.path.join(train_dir, "Fire")
train_no_fire_dir = os.path.join(train_dir, "No Fire")

validation_fire_dir = os.path.join(validation_dir, "Fire")
validation_no_fire_dir = os.path.join(validation_dir, "No Fire")

test_fire_dir = os.path.join(test_dir, "Fire")
test_no_fire_dir = os.path.join(test_dir, "No Fire")

# Create directories if they don't exist
for directory in [train_fire_dir, train_no_fire_dir, validation_fire_dir, validation_no_fire_dir, test_fire_dir, test_no_fire_dir]:
    os.makedirs(directory, exist_ok=True)

# Split data into train, validation, and test
for class_dir in [fire_dir, no_fire_dir]:
    images = os.listdir(class_dir)
    np.random.shuffle(images)
    
    train_split = int(0.7 * len(images))
    validation_split = int(0.15 * len(images))
    
    train_images = images[:train_split]
    validation_images = images[train_split:train_split + validation_split]
    test_images = images[train_split + validation_split:]
    
    for img in train_images:
        source_path = os.path.join(class_dir, img)
        destination_path = os.path.join(train_dir, os.path.join(class_dir.split("\\")[-1], img))
        shutil.copy(source_path, destination_path)
    
    for img in validation_images:
        source_path = os.path.join(class_dir, img)
        destination_path = os.path.join(validation_dir, os.path.join(class_dir.split("\\")[-1], img))
        shutil.copy(source_path, destination_path)
    
    for img in test_images:
        source_path = os.path.join(class_dir, img)
        destination_path = os.path.join(test_dir, os.path.join(class_dir.split("\\")[-1], img))
        shutil.copy(source_path, destination_path)

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary'
)

# Hyperparameter tuning using keras-tuner
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('conv1_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('conv2_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('conv3_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('dense_units', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=2,  # Adjust the number of trials as needed
    directory='hyperparameter_tuning',
    project_name='wildfire_classification'
)

# Train the tuner
tuner.search(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Get the best model and summary of hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test Accuracy: {test_acc}")

# Confusion Matrix and Classification Report
predictions = best_model.predict(test_generator, steps=test_generator.samples // batch_size)
rounded_predictions = np.round(predictions)

#conf_matrix = confusion_matrix(test_generator.classes, rounded_predictions)
classification_report_str = classification_report(test_generator.classes, rounded_predictions)

print("Confusion Matrix:")
#print(conf_matrix)
print("\nClassification Report:")
print(classification_report_str)

# Visualize training history
epochs = range(1, 20 + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, label='Training Accuracy')
plt.plot(epochs, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, label='Training Loss')
plt.plot(epochs, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
