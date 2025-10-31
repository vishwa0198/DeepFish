import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 🔹 Define dataset and model base path
BASE_DIR = r"C:\Guvi\Fish Classification"
MODELS_DIR = os.path.join(BASE_DIR, r"C:\Guvi\Fish Classification\Tranied models")
TEST_DIR = os.path.join(BASE_DIR, r"C:\Guvi\Fish Classification\data\test")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# 🔹 Load test data
datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
class_names = list(test_gen.class_indices.keys())

# 🔹 Evaluation Function
def evaluate_model(model_path, test_gen, class_names):
    print(f"\n📊 Evaluating: {model_path}")
    model = load_model(model_path)
    
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"✅ Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    preds = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    # Save Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    report_name = os.path.basename(model_path).replace('.h5', '_report.csv')
    df_report.to_csv(os.path.join(REPORTS_DIR, report_name))

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {os.path.basename(model_path)}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, os.path.basename(model_path).replace('.h5', '_cm.png')))
    plt.close()
    return acc, loss

# 🔹 Loop through all subfolders
results = []
for folder in os.listdir(MODELS_DIR):
    subdir = os.path.join(MODELS_DIR, folder)
    if os.path.isdir(subdir):
        for file in os.listdir(subdir):
            if file.endswith(".h5"):
                model_path = os.path.join(subdir, file)
                acc, loss = evaluate_model(model_path, test_gen, class_names)
                results.append({"Model": folder, "Accuracy": acc, "Loss": loss})

# 🔹 Save model comparison summary
df_summary = pd.DataFrame(results)
df_summary.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
print("\n✅ All model evaluations completed and saved in /reports/")
