import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import argparse

def main(class_0_file, class_1_file):
    # Step 1: Load the two JSONL files, one for each class
    class_0_data = pd.read_json(class_0_file, lines=True)
    class_1_data = pd.read_json(class_1_file, lines=True)

    # Step 2: Extract the 'text' field and assign labels
    class_0_texts = class_0_data['text']
    class_1_texts = class_1_data['text']

    # Assign labels: 0 for class_0 and 1 for class_1
    class_0_labels = [0] * len(class_0_texts)
    class_1_labels = [1] * len(class_1_texts)

    # Step 3: Combine the two classes
    texts = pd.concat([class_0_texts, class_1_texts])
    labels = class_0_labels + class_1_labels

    # Step 4: Convert texts to Bag of Words representation
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    # Step 5: Train a classifier on all data
    model = LogisticRegression(n_jobs=38, verbose=1)

    print("Start training...")
    model.fit(X, labels)
    print("End training.")

    # Save the model and vectorizer with hardcoded filenames
    joblib.dump(model, 'BoW.joblib')
    print("Model saved to BoW.joblib")

    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Vectorizer saved to vectorizer.joblib")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model on two JSONL files.")
    parser.add_argument("class_0_file", help="Path to the JSONL file for class 0")
    parser.add_argument("class_1_file", help="Path to the JSONL file for class 1")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args.class_0_file, args.class_1_file)
