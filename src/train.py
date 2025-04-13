import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from keras import layers, models

# --------- Setup Directories ---------
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_dir, 'results')
reports_dir = os.path.join(results_dir, 'raports')
visuals_dir = os.path.join(results_dir, 'visualisation')
models_dir = os.path.join(results_dir, 'models')

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(visuals_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --------- Step 1: Load Dataset ---------
def load_images_from_folder(folder_path, label, img_size=(299, 299)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels

# Dataset setup
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(project_dir, "COVID-19_Radiography_Dataset")
img_size = (299, 299)

# Define correct paths to "images" folders
covid_path = os.path.join(dataset_path, "COVID", "images")
normal_path = os.path.join(dataset_path, "Normal", "images")

# Load data
covid_images, covid_labels = load_images_from_folder(covid_path, 1, img_size)
normal_images, normal_labels = load_images_from_folder(normal_path, 0, img_size)

data = covid_images + normal_images
labels = covid_labels + normal_labels

# Convert to numpy arrays
data = np.array(data).reshape(-1, img_size[0], img_size[1], 1) / 255.0
labels = np.array(labels)

# --------- Step 2: Train-Test Split ---------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# --------- Step 3: Build CNN Model ---------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model.summary()

# --------- Step 4: Train Model ---------
history = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test), batch_size=32)
model.save(os.path.join(project_dir, "results", "models", "covid_cnn_model.h5"))

# --------- Step 5: Evaluation ---------
y_probs = model.predict(X_test)
y_pred = (y_probs > 0.5).astype(int).flatten()

target_names = ['Normal', 'COVID']
report = classification_report(y_test, y_pred, target_names=target_names)
print("\nClassification Report:")
print(report)

report_path = os.path.join(reports_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
confusion_path = os.path.join(visuals_dir, 'confusion_matrix.png')
plt.savefig(confusion_path)
plt.close()

# --------- Step 6: Visualize Training Progress ---------
# Accuracy & Loss
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

acc_loss_path = os.path.join(visuals_dir, 'accuracy_loss.png')
plt.savefig(acc_loss_path)
plt.close()

# Precision & Recall
plt.figure(figsize=(14, 5))

# Precision
plt.subplot(1, 2, 1)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Precision over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Recall
plt.subplot(1, 2, 2)
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Recall over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

pr_re_path = os.path.join(visuals_dir, 'precision_recall.png')
plt.savefig(pr_re_path)
plt.close()