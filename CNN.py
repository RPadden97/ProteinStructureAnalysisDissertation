import numpy as np
import pandas as pd
from ast import literal_eval
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, InputLayer, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pandarallel import pandarallel
import json
from tensorflow.keras.callbacks import ModelCheckpoint

pandarallel.initialize(progress_bar=True)

def load_file(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    data['PRIMARY'] = data['PRIMARY'].apply(vectorize_sequence)
    data['EVOLUTIONARY'] = data['EVOLUTIONARY'].parallel_apply(string_to_float_array)
    data['TERTIARY'] = data['TERTIARY'].parallel_apply(string_to_float_array)

    return data

def preprocess_data(data, parameter, label_column, consistent_shape=(350, 25)):
    """
    Preprocess the data to extract a single feature parameter and labels.
    """

    if parameter == 'PRIMARY':
        feature = data[parameter].apply(np.array)
    elif parameter == 'EVOLUTIONARY':
        feature = data[parameter].apply(np.array).apply(lambda x: pad_or_truncate(x, (750,21)))
    else:
        print('in hereeeeeee')
        feature = data[parameter].apply(np.array).parallel_apply(lambda x: pad_or_truncate(x, (3,750)))

    feature = np.stack(feature)  # Combine into a single array
    
    labels = data[label_column]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)  # Encode categorical labels
    labels = to_categorical(labels)  # Convert to one-hot encoding
    
    return feature, labels, label_encoder.classes_

# Function to pad or truncate a sequence to consistent_shape
def pad_or_truncate(sequence, shape):
    array = np.array(sequence)
    # Pad or truncate rows
    array = np.pad(array, ((0, max(0, shape[0] - array.shape[0])), (0, max(0, shape[1] - array.shape[1]))), mode='constant')
    return array[:shape[0], :shape[1]]  # Crop to the target shape if too large

def string_to_float_array(string):
    """
    Converts a string representation of an array of arrays of strings to an array of arrays of floats.

    Args:
        string (str): The string representation of the array of arrays.

    Returns:
        list: A nested list of floats.
    """

    # Convert the string to a Python object
    array_of_arrays = literal_eval(string)
    
    # Convert each string element to a float
    float_array = [[float(value) for value in inner_array] for inner_array in array_of_arrays]
    
    return float_array

def vectorize_sequence(sequence, max_length=700):
    """
    Vectorize an amino acid sequence using embedding encoding and pad/truncate to a specified length.

    Parameters:
    - sequence (str): The amino acid sequence to be vectorized.
    - max_length (int): The desired length for the vectorized sequence. Default is 350.

    Returns:
    - np.ndarray: A (max_length, 25) size vector representation of the sequence.
    """

    # Amino acid embedding dictionary
    embedding = {
        'L': 1, 'A': 2, 'G': 3, 'V': 4, 'E': 5, 'S': 6, 'D': 7, 'T': 8,
        'I': 9, 'K': 10, 'R': 11, 'P': 12, 'N': 13, 'F': 14, 'Q': 15,
        'Y': 16, 'H': 17, 'M': 18, 'W': 19, 'C': 20, 'X': 21, 'Z': 22,
        'U': 23, 'B': 24, 'O': 25
    }

    # Initialize the vector matrix
    vectorized = np.zeros((max_length, 25))

    # Process the sequence
    for i, amino_acid in enumerate(sequence[:max_length]):  # Truncate if sequence is longer than max_length
        if amino_acid in embedding:
            vectorized[i, embedding[amino_acid] - 1] = 1  # One-hot encoding

    return vectorized

def create_cnn_model(input_shape, num_classes):
    """
    A simple CNN model.
    """
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(parameter, train_file, val_file, test_file, label_column='LABEL'):
    """
    Train and evaluate a model for a specific parameter.
    """
    print(f"Training model for parameter: {parameter}")
    
    # Preprocess data
    train_features, train_labels, class_names = preprocess_data(train_file, parameter, label_column)
    val_features, val_labels, _ = preprocess_data(val_file, parameter, label_column)
    test_features, test_labels, _ = preprocess_data(test_file, parameter, label_column)
    
    # Get input shape and number of classes
    input_shape = train_features[0].shape + (1,)  # Add channel dimension
    num_classes = train_labels.shape[1]
    
    # Reshape features to add channel dimension
    train_features = train_features[..., np.newaxis]
    val_features = val_features[..., np.newaxis]
    test_features = test_features[..., np.newaxis]
    
    # Create model
    model = create_cnn_model(input_shape, num_classes)

    # Define checkpoint to save the best model during training
    checkpoint = ModelCheckpoint(f'cnn_model_{parameter}.keras', monitor='val_accuracy', save_best_only=True, mode='max')

    # Train model
    history = model.fit(train_features, train_labels, epochs=20, batch_size=32, validation_data=(val_features, val_labels), callbacks=[checkpoint])
    
    # Evaluate model
    loss, accuracy = model.evaluate(test_features, test_labels)
    print(f"Test Accuracy for {parameter}: {accuracy * 100:.2f}%")

    # Save the training history
    with open(f'cnn_model_{parameter}_history.json', 'w') as f:
        json.dump(history.history, f)

    print(f"Model and training history saved for cnn_model{parameter}!")
    
    return accuracy

# Paths to datasets
train_file = './casp7/training_100.csv'
val_file = './casp7/validation.csv'
test_file = './casp7/testing.csv'

# Load files
training_data = load_file(train_file)
val_data = load_file(val_file)
test_data = load_file(test_file)

# Parameters to test
parameters = ['PRIMARY', 'EVOLUTIONARY', 'TERTIARY']
results = {}

# Train and evaluate for each parameter
for param in parameters:
    results[param] = train_and_evaluate(param, training_data, val_data, test_data)

# Print results
print("\nModel Performance:")
for param, accuracy in results.items():
    print(f"{param}: {accuracy * 100:.2f}%")
