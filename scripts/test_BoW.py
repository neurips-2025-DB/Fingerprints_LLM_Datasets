import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import argparse

def main(class_0_test_file, class_1_test_file):
    # Step 1: Load the two test JSONL files, one for each class
    class_0_test_data = pd.read_json(class_0_test_file, lines=True)
    class_1_test_data = pd.read_json(class_1_test_file, lines=True)

    # Step 2: Extract the 'text' field and assign labels
    class_0_test_texts = class_0_test_data['text']
    class_1_test_texts = class_1_test_data['text']

    # Assign labels: 0 for class_0 and 1 for class_1
    class_0_test_labels = [0] * len(class_0_test_texts)
    class_1_test_labels = [1] * len(class_1_test_texts)

    # Step 3: Combine the two classes
    test_texts = pd.concat([class_0_test_texts, class_1_test_texts])
    test_labels = class_0_test_labels + class_1_test_labels

    # Step 4: Load the vectorizer and transform the test texts
    vectorizer = joblib.load('vectorizer.joblib')
    X_test = vectorizer.transform(test_texts)

    # Step 5: Load the trained model and make predictions
    model = joblib.load('BoW.joblib')
    predictions = model.predict(X_test)

    # Step 6: Evaluate the model using the true test labels
    accuracy = accuracy_score(test_labels, predictions)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate a trained Logistic Regression model using test JSONL files.")
    parser.add_argument("class_0_test_file", help="Path to the test JSONL file for class 0")
    parser.add_argument("class_1_test_file", help="Path to the test JSONL file for class 1")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args.class_0_test_file, args.class_1_test_file)
