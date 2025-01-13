#dependencies:
# pip install tensorflow keras-tuner scikit-learn matplotlib seaborn

# For using Keras Tuner for hyperparameter tuning install following:
# !pip install -q keras-tuner

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import keras_tuner as kt

##############################################################################
# 1. DATA LOADING & PREPROCESSING
##############################################################################

# -----------------------------------------------------
# Example: Using CIFAR-10 from Keras datasets
# -----------------------------------------------------
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalize pixel values to [0, 1]
x_train /= 255.0
x_test /= 255.0

# Convert class vectors to one-hot encodings
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# -----------------------------------------------------
# Split into training/validation sets
# -----------------------------------------------------
# Typically, we take ~10-20% of training as validation
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42
)

print("Training Set   :", x_train.shape, y_train.shape)
print("Validation Set :", x_val.shape, y_val.shape)
print("Test Set       :", x_test.shape, y_test.shape)

# -----------------------------------------------------
# Data Augmentation
# -----------------------------------------------------
# Use ImageDataGenerator for real-time augmentation
# (rotation, shift, flip, etc.)
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator()  # No augmentation for validation/test

train_generator = train_datagen.flow(
    x_train, y_train, batch_size=64, shuffle=True
)
val_generator = val_datagen.flow(
    x_val, y_val, batch_size=64, shuffle=False
)


##############################################################################
# 2. MODEL DEVELOPMENT
#    (You can tweak the architecture, # of layers, filters, etc.)
##############################################################################

def build_cnn_model(
        input_shape=(32, 32, 3),
        num_classes=10,
        l2_reg=1e-4,
        dropout_rate=0.5
):
    """
    Build a simple CNN model with L2 regularization and dropout.
    """
    weight_decay = regularizers.l2(l2_reg)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=weight_decay, input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=weight_decay),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=weight_decay),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=weight_decay),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=weight_decay),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=weight_decay),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=weight_decay),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


##############################################################################
# (OPTIONAL) TRANSFER LEARNING EXAMPLE (COMMENTED OUT)
# Uncomment this section to use a pre-trained model like MobileNetV2.
##############################################################################
'''
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def build_transfer_model(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze all convolutional layers to preserve pre-trained weights
    for layer in base_model.layers:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
'''

##############################################################################
# 3. REGULARIZATION: DROPOUT, L2, EARLY STOPPING
#    Already integrated into the build_cnn_model function.
##############################################################################

# We'll define a basic model using our build_cnn_model function.
# Later, we will do hyperparameter tuning to find optimal settings.
model = build_cnn_model(l2_reg=1e-4, dropout_rate=0.4)


##############################################################################
# 4. HYPERPARAMETER TUNING
#    We'll demonstrate two approaches using Keras Tuner:
#    - RandomSearch
#    - Hyperband or Bayesian could also be used,
#      but we stick to RandomSearch to illustrate.
##############################################################################

def model_builder(hp):
    """
    This function is used by Keras Tuner to build
    a model with hyperparameters chosen from ranges.
    """
    # Hyperparameter choices:
    hp_l2 = hp.Choice('l2_reg', values=[1e-4, 1e-3, 1e-2])
    hp_dropout = hp.Choice('dropout_rate', values=[0.3, 0.4, 0.5])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    model = build_cnn_model(l2_reg=hp_l2, dropout_rate=hp_dropout)

    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# -----------------------------------------------------
# Random Search with Keras Tuner
# -----------------------------------------------------
tuner = kt.RandomSearch(
    model_builder,
   objective='val_accuracy',
    max_trials=5,  # Increase for more thorough search
    executions_per_trial=1,
    directory='my_tuner_dir',
    project_name='cifar10_random_search'
)

# Uncomment below lines to perform the hyperparameter search
# tuner.search(
#     train_generator,
#     validation_data=val_generator,
#     epochs=5,
#     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
# )

# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print("Best hyperparameters found:", best_hps.values)

##############################################################################
# 5. FINAL MODEL TRAINING & EVALUATION
##############################################################################

# If you used tuner, you could retrieve best hyperparams:
# l2_reg_best = best_hps.get('l2_reg')
# dropout_best = best_hps.get('dropout_rate')
# lr_best = best_hps.get('learning_rate')
# final_model = build_cnn_model(l2_reg=l2_reg_best, dropout_rate=dropout_best)
# final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_best),
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy'])

final_model = build_cnn_model(l2_reg=1e-4, dropout_rate=0.4)
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early Stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = final_model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------------------------------
# Evaluate on test set
# -----------------------------------------------------
test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=0)
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_acc))


##############################################################################
# 6. MODEL EVALUATION & RESULTS ANALYSIS
##############################################################################

# -------------------
# Plot Training Curves
# -------------------
def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()


plot_training_curves(history)

# -------------------
# Confusion Matrix
# -------------------
y_pred_probs = final_model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix on Test Data")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -------------------
# Classification Report
# -------------------
print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4))
