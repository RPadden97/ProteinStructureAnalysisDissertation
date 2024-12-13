from tensorflow.keras.models import load_model
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def load_model_and_history(model_name):
    # Load the model
    model = load_model(f'{model_name}.keras')

    # Load the history
    with open(f'{model_name}_history.json', 'r') as f:
        history = json.load(f)

    return model, history

# Load models and histories
primary_model, primary_history = load_model_and_history('cnn_model_primary')
evolutionary_model, evolutionary_history = load_model_and_history('cnn_model_evolutionary')
tertiary_model, tertiary_history = load_model_and_history('cnn_model_tertiary')
histories=[primary_history,evolutionary_history,tertiary_history]
labels=['HYDROLASE', 'TRANSFERASE', 'OXIDOREDUCTASE', 'LYASE', 'IMMUNE SYSTEM', 'TRANSCRIPTION', 'TRANSPORT PROTEIN', 'SIGNALING PROTEIN', 'ISOMERASE', 'VIRAL PROTEIN', 'LIGASE', 'PROTEIN BINDING', 'DNA','STRUCTURAL GENOMICS', 'MEMBRANE PROTEIN', 'TRANSFERASE INHIBITOR', 'DNA BINDING PROTEIN','RIBOSOME','METAL BINDING PROTEIN','NONE',]


def plot_accuracy_bar_chart(accuracies, labels):
    plt.figure(figsize=(8, 6))
    plt.bar(labels, accuracies, alpha=0.7)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_confusion_matrix(true_labels, predictions, model_name, classes):
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, colorbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

def plot_roc_curve(true_labels, probabilities, model_name, num_classes):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(true_labels[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_training_loss_curves(histories, labels):
    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(history['loss'], label=f'{label} (Training Loss)')
        plt.plot(history['val_loss'], '--', label=f'{label} (Validation Loss)')
    
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_normalized_accuracy_heatmap(accuracies, labels, num_classes):
    normalized_accuracies = np.array(accuracies) / np.max(accuracies, axis=1)[:, None]
    plt.figure(figsize=(10, 6))
    plt.imshow(normalized_accuracies, cmap='viridis', aspect='auto')
    plt.colorbar(label='Normalized Accuracy')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(num_classes), [f'Class {i}' for i in range(num_classes)])
    plt.title('Normalized Accuracy Heatmap')
    plt.xlabel('Models')
    plt.ylabel('Classes')
    plt.show()


# 1. Bar Chart for Accuracy Comparison
plot_accuracy_bar_chart([0.0408, 0.1735, 0.1837], ['Primary', 'Evolutionary', 'Tertiary'])

# 2. Confusion Matrices
# plot_confusion_matrix(y_test_primary, y_pred_primary, 'Primary', class_names)

# 3. ROC Curves
# plot_roc_curve(y_test_primary_onehot, y_prob_primary, 'Primary', num_classes=19)

# 4. Loss Curves During Training
plot_training_loss_curves(histories, ['Primary', 'Evolutionary', 'Tertiary'])

# 5. Normalized Accuracy Heatmap
accuracies = [
    primary_history['accuracy'],  # Primary
    evolutionary_history['accuracy'],  # Evolutionary
    tertiary_history['accuracy'],   # Tertiary
]
print(accuracies)
plot_normalized_accuracy_heatmap(accuracies, labels, num_classes=3)