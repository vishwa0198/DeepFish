import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd

# Paths
base_model_dir = r"C:\Guvi\Fish Classification\Tranied models"
test_dir = r"C:\Guvi\Fish Classification\data\test"
report_path = r"C:\Guvi\Fish Classification\model_test_report.csv"

# Load test data
IMG_SIZE = (224, 224)  # adjust if needed
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Prepare results storage
results = []

# Loop through each model folder
for model_name in os.listdir(base_model_dir):
    model_folder = os.path.join(base_model_dir, model_name)
    
    if os.path.isdir(model_folder):
        # Find the .h5 model file
        model_files = [f for f in os.listdir(model_folder) if f.endswith(".h5")]
        if not model_files:
            continue

        model_path = os.path.join(model_folder, model_files[0])
        print(f"\n📊 Evaluating: {model_path}")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Evaluate
        loss, acc = model.evaluate(test_generator, verbose=0)
        print(f"✅ {model_name} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")

        # Store results
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Loss": loss
        })

# Save all results to CSV
os.makedirs(os.path.dirname(report_path), exist_ok=True)
pd.DataFrame(results).to_csv(report_path, index=False)

print("\n✅ All model evaluations completed.")
print(f"📁 Results saved to: {report_path}")
