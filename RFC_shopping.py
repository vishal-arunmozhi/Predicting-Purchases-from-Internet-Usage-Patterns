import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

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

    # sampling the data to deal with the imbalance of the classes
    # Random Under Sampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    """# Random Over Sampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    # SMOTE
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)"""
    
    # taking the trained model and making predictions
    model = classifier_model(X_resampled, y_resampled)
    predictions = model.predict(X_test)

    # calculating sensitivity and specificity values
    sensitivity, specificity = evaluate(y_test, predictions)

    # calculating the Youden's J statistic
    youden_j = sensitivity + specificity - 1

    # Printing the results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"Youden's J Statistic: {youden_j:.3f}")


# function takes in the filename and returns features and labels
def load_data(filename):
    features = []
    labels = []
    
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
        'Returning_Visitor': 1
    }

    truth_index = {
        'FALSE': 0,
        'TRUE': 1
    }

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # skipping the first row since it contains the column names
        next(csvreader)

        for row in csvreader:
            # converting categorical month data into numerical data, row[10] refers to month column in each row
            month_name = row[10]
            month_index = month_to_index.get(month_name, -1)
            row[10] = month_index

            # converting visitortype into numerical data, row[15]
            visitortype = row[15]
            visitor_index = visitortype_index.get(visitortype, -1)
            row[15] = visitor_index

            # converting weekend column into numerical data, row[16]
            weekend = row[16]
            weekend_index = truth_index.get(weekend, -1)
            row[16] = weekend_index

            #converting revenue column into numerical data, row[17]
            revenue = row[17]
            revenue_index = truth_index.get(revenue, -1)
            row[17] = revenue_index

            features.append([int(row[0]), float(row[1]), int(row[2]), 
                             float(row[3]), int(row[4]), float(row[5]), 
                             float(row[6]), float(row[7]), float(row[8]), 
                             float(row[9]), int(row[10]), int(row[11]), 
                             int(row[12]), int(row[13]), int(row[14]), 
                             int(row[15]), int(row[16])
                             ])
            
            labels.append(int(row[17]))

    return (features, labels)


# function takes features and labels as inputs and returns the trained model
def classifier_model(features, labels):
    # Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    """# KNN algorithm
    k = 1
    model = KNeighborsClassifier(n_neighbors=k)

    # Gradient Boosting Classifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)"""
    
    # training the model
    model.fit(features, labels)

    return model


# function takes labels and predictions as input and returns specificity and sensitivity
def evaluate(labels, predictions):
    negative_labels = labels.count(0)
    positive_labels = labels.count(1)

    negative_labels_pred = 0 # total number of negative labels correctly predicted
    positive_labels_pred = 0 # total number of positive labels correctly predicted
    
    for index, label in enumerate(labels):
        if label == 0 and label == predictions[index]:
            negative_labels_pred += 1
        elif label == 1 and label == predictions[index]:
            positive_labels_pred += 1

    specificity = negative_labels_pred/negative_labels
    sensitivity = positive_labels_pred/positive_labels
    
    return (sensitivity, specificity)
    

if __name__ == "__main__":
    main()
