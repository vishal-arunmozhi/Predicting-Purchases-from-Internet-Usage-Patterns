import sys
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from keras import models, layers, optimizers

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")
        
    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=42
    )

    # code to sample the data to deal with the amount of imbalance present
    # Random Over Sampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    """#Random Under Sampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # SMOTE
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)"""
    
    # Train model and make predictions
    model = nn_model(X_resampled, y_resampled)

    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1) 
    
    # calculating sensitivity and specificity values
    sensitivity, specificity = evaluate(y_test, predictions)

    # calculating the Youden's J statistic
    youden_j = sensitivity + specificity - 1

    # Printing results
    print(f"Correct: {(y_test == np.round(predictions)).sum()}")
    print(f"Incorrect: {(y_test != np.round(predictions)).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"Youden's J Statistic: {youden_j:.3f}")


# function takes in the filename and returns features and labels
def load_data(filename):
    dataset = pd.read_csv(filename)

    month_to_index = {
        'Jan': 0,
        'Feb': 1,
        'Mar': 2,
        'Apr': 3,
        'May': 4,
        'June': 5,
        'Jul': 6,
        'Aug': 7,
        'Sep': 8,
        'Oct': 9,
        'Nov': 10,
        'Dec': 11
    }

    visitortype_index = {
        'New_Visitor': 0,
        'Returning_Visitor': 1,
        'Other': -1
    }

    truth_index = {
        False: 0,
        True: 1
    }

    # convert categorical data to numerical data
    dataset['Month'].replace(month_to_index, inplace=True)
    dataset['VisitorType'].replace(visitortype_index, inplace=True)
    dataset['Weekend'].replace(truth_index, inplace=True)
    dataset['Revenue'].replace(truth_index, inplace=True)

    # Z-score normalization of first 9 feature columns
    scaler = StandardScaler()
    for col in dataset.columns[:9]:
        dataset[col] = scaler.fit_transform(dataset[[col]])

    # one-hot encoding feature columns 11 to 14
    for  col in dataset.columns[11:15]:
        dataset = pd.get_dummies(dataset, columns=[col], prefix=[col], dtype=int)  
    
    labels = np.array(dataset.pop('Revenue'))
    features = np.array(dataset)

    return features, labels


# function takes features and labels as inputs and returns the trained model
def nn_model(features, labels):
    model = models.Sequential()

    model.add(layers.Dense(32, input_shape=(63,), activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    # custom metric function to calculate Youden Index as a metric
    def youden_index(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        negative_labels = K.sum(1 - y_true)
        positive_labels = K.sum(y_true)

        negative_labels_pred = K.sum(tf.cast((1 - y_true) * (1 - K.round(y_pred)), tf.float32))
        positive_labels_pred = K.sum(tf.cast(y_true * K.round(y_pred), tf.float32))

        specificity_metric = negative_labels_pred/(negative_labels + K.epsilon())
        sensitivity_metric = positive_labels_pred/(positive_labels + K.epsilon())

        youden_index = sensitivity_metric + specificity_metric - 1

        return youden_index

    model.compile(optimizer=optimizers.Adam(0.001), loss='binary_crossentropy', metrics=[youden_index])
    
    # Giving a slightly higher weight for the minority class even after data sampling has been done
    class_weights = {0: 1, 1: 1.2}
    history = model.fit(features, labels, batch_size=32, class_weight=class_weights, epochs=10, verbose=2)
    plot_loss(history)

    return model


# function takes the history variable as input and plots training loss and youden index
def plot_loss(history):
    training_loss = history.history['loss']
    youden_index = history.history['youden_index']

    epochs = list(range(1, len(training_loss) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, youden_index, label='Youden Index')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# function takes labels and predictions as input and returns specificity and sensitivity
def evaluate(labels, predictions):
    negative_labels = np.sum(1 - labels)
    positive_labels = np.sum(labels)

    negative_labels_pred = np.sum((1 - labels) * (1 - np.round(predictions))) # total number of negative labels correctly predicted
    positive_labels_pred = np.sum(labels * np.round(predictions)) # total number of positive labels correctly predicted

    specificity = negative_labels_pred/negative_labels
    sensitivity = positive_labels_pred/positive_labels
    
    return (sensitivity, specificity)
    

if __name__ == "__main__":
    main()
