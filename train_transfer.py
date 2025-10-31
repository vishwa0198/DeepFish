import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_loader import load_data

# ===========================
# 1. Config
# ===========================
BASE_DIR = r'data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Choose which pre-trained model to use 👇
MODEL_NAME = 'EfficientNetB0'   # Change to: 'VGG16', 'MobileNetV2', 'InceptionV3', 'EfficientNetB0'

# ===========================
# 2. Load Dataset
# ===========================
train_gen, val_gen, test_gen = load_data(BASE_DIR, IMG_SIZE, BATCH_SIZE)
num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

# ===========================
# 3. Model Builder
# ===========================
def build_transfer_model(model_name, input_shape=(224,224,3), num_classes=5):
    preprocess_func = None

    if model_name == 'VGG16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_func = preprocess_input

    elif model_name == 'ResNet50':
        from tensorflow.keras.applications.resnet import preprocess_input
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_func = preprocess_input

    elif model_name == 'MobileNetV2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_func = preprocess_input

    elif model_name == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_func = preprocess_input

    elif model_name == 'EfficientNetB0':
        from tensorflow.keras.applications.efficientnet import preprocess_input
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_func = preprocess_input

    else:
        raise ValueError("Unsupported model name")

    # Freeze base layers initially
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model, preprocess_func


# ===========================
# 4. Build and Train
# ===========================
model, base_model, preprocess_func = build_transfer_model(
    MODEL_NAME, input_shape=(224,224,3), num_classes=num_classes
)
model.summary()

# Ensure save directory exists
os.makedirs('saved_models', exist_ok=True)

# Callbacks
callbacks = [
    ModelCheckpoint(f'saved_models/{MODEL_NAME}_best.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train the model (initial training)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===========================
# 5. Fine-tuning (optional)
# ===========================
print("\n🔧 Fine-tuning top layers...")

# Unfreeze the base model
base_model.trainable = True

# Keep lower layers frozen
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)

# ===========================
# 6. Evaluate on Test Set
# ===========================
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy ({MODEL_NAME}): {test_acc * 100:.2f}%")

# ===========================
# 7. Plot Training Curves
# ===========================
plt.figure(figsize=(10, 4))

# Combine histories for full curve
train_acc = history.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
train_loss = history.history['loss'] + history_finetune.history['loss']
val_loss = history.history['val_loss'] + history_finetune.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.legend()
plt.title(f'{MODEL_NAME} Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
plt.title(f'{MODEL_NAME} Loss')

plt.show()

# ===========================
# 8. Save Final Model
# ===========================
final_path = f"saved_models/{MODEL_NAME}_final.h5"
model.save(final_path)
print(f"\n💾 Model saved successfully at: {final_path}")
