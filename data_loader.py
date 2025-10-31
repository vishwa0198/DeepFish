from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_data(base_dir, img_size=(224, 224), batch_size=32):
    """
    Loads and preprocesses the dataset using ImageDataGenerator.
    """

    # Define directory paths
    train_dir = os.path.join(base_dir, r'C:\Guvi\Fish Classification\data\train')
    val_dir = os.path.join(base_dir, r'C:\Guvi\Fish Classification\data\val')
    test_dir = os.path.join(base_dir, r'C:\Guvi\Fish Classification\data\test')

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and test data should NOT be augmented
    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Load images from directories
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Don't shuffle test data
    )

    return train_data, val_data, test_data


# Run directly (for testing)
if __name__ == "__main__":
    base_dir = r'data'
    train_data, val_data, test_data = load_data(base_dir, img_size=(224, 224), batch_size=32)

    # Print some dataset info
    print("\n✅ Dataset Loaded Successfully!")
    print("Classes:", train_data.class_indices)
    print("Number of training samples:", train_data.samples)
    print("Number of validation samples:", val_data.samples)
    print("Number of test samples:", test_data.samples)
