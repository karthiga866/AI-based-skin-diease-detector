import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os

# Load Pretrained Model (Feature Extractor)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Add Custom Layers for Classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(5, activation='softmax')  # Change '5' to the number of skin diseases

# Build Model
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory('dataset/', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
val_generator = train_datagen.flow_from_directory('dataset/', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')

# Train Model
epochs = 10
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save Model
model.save("skin_disease_model.h5")

# Plot Training History
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction Function
def predict_skin_disease(image_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_names = list(train_generator.class_indices.keys())
    return class_names[np.argmax(prediction)]

# Example Prediction
print(predict_skin_disease('test_image.jpg'))
