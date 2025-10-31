import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import load_data

# ===========================
# 1. Load Dataset
# ===========================
train_gen, val_gen, test_gen = load_data(base_dir=r'data', img_size=(224, 224), batch_size=32)

num_classes = len(train_gen.class_indices)
print("Number of classes:", num_classes)
print("Classes:", train_gen.class_indices)

# ===========================
# 2. Define CNN Model (from scratch)
# ===========================
def build_cnn(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


model = build_cnn(num_classes=num_classes)
model.summary()

# ===========================
# 3. Compile Model
# ===========================
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===========================
# 4. Add Callbacks
# ===========================
checkpoint = ModelCheckpoint(
    "saved_models/cnn_baseline_best.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

callbacks = [checkpoint, early_stop]

# ===========================
# 5. Train Model
# ===========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=callbacks
)

# ===========================
# 6. Evaluate Model
# ===========================
test_loss, test_acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# ===========================
# 7. Plot Training Curves
# ===========================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# ===========================
# 8. Save Final Model
# ===========================
model.save("saved_models/cnn_baseline_final.h5")
print("Model saved at saved_models/cnn_baseline_final.h5")
