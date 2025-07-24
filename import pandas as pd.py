import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os

def predict_titanic_survival():
    print("=== Titanic Survival Prediction Started ===")

    dataset = None
    filenames = ['Titanic-Dataset.csv - Titanic-Dataset.csv', 'Titanic-Dataset.csv']

    # Try loading known filenames
    for name in filenames:
        try:
            dataset = pd.read_csv(name)
            print(f"Loaded dataset: '{name}', shape: {dataset.shape}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Format issue with '{name}': {e}")

    # Try auto-detecting dataset if not found
    if dataset is None:
        print("Auto-detecting Titanic dataset in current directory...")
        for fname in os.listdir('.'):
            if "titanic" in fname.lower() and fname.endswith(".csv"):
                try:
                    dataset = pd.read_csv(fname)
                    print(f"Auto-loaded dataset: '{fname}', shape: {dataset.shape}")
                    break
                except Exception as e:
                    print(f"Could not read '{fname}': {e}")

    if dataset is None:
        print("ERROR: Titanic dataset not found. Please upload the correct CSV file.")
        return

    print("\n--- First 5 Rows ---")
    print(dataset.head())

    print("\n--- Missing Value Report ---")
    print(dataset.isnull().sum())

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    target = 'Survived'

    # Impute missing Age and Fare
    if 'Age' in dataset.columns:
        dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    if 'Fare' in dataset.columns and dataset['Fare'].isnull().any():
        dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)

    # Encode 'Sex'
    if 'Sex' in dataset.columns:
        dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
        dataset['Sex'].fillna(dataset['Sex'].mode()[0], inplace=True)

    # Check if all required features are present
    for col in features:
        if col not in dataset.columns:
            print(f"ERROR: Missing column '{col}' in dataset.")
            return

    # Drop remaining missing values in selected columns
    before_drop = dataset.shape[0]
    dataset.dropna(subset=features + [target], inplace=True)
    after_drop = dataset.shape[0]
    if before_drop != after_drop:
        print(f"Dropped {before_drop - after_drop} incomplete rows.")

    X = dataset[features]
    y = dataset[target]

    print(f"\nFeature shape: {X.shape}, Target shape: {y.shape}")
    print("\n--- First 5 Feature Rows ---")
    print(X.head())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("\nData split: 80% train, 20% test")

    # Train logistic regression
    model = LogisticRegression(solver='liblinear', random_state=42)
    print("\nTraining logistic regression model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    # Coefficients
    print("\n--- Model Weights & Bias ---")
    for feature, weight in zip(features, model.coef_[0]):
        print(f"{feature}: {weight:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")

    # Sample passengers
    sample_data = pd.DataFrame([
        [1, 1, 25, 0, 0, 100],
        [3, 0, 60, 1, 0, 15],
        [2, 0, 20, 0, 0, 25]
    ], columns=features)

    print("\n--- Sample Passenger Predictions ---")
    probs = model.predict_proba(sample_data)[:, 1]
    preds = model.predict(sample_data)

    for idx, (row, prob, pred) in enumerate(zip(sample_data.itertuples(index=False), probs, preds), 1):
        outcome = "Survived" if pred == 1 else "Did Not Survive"
        sex = "Female" if row.Sex == 1 else "Male"
        print(f"\nPassenger {idx}:")
        print(f"  Class: {row.Pclass}, Sex: {sex}, Age: {row.Age}")
        print(f"  Prediction: {outcome}, Probability: {prob * 100:.2f}%")

    # Test set results
    test_results = pd.DataFrame({
        'Actual': y_test.reset_index(drop=True),
        'Predicted': y_pred,
        'Prob_Survived': model.predict_proba(X_test)[:, 1]
    })

    test_results['Actual_Label'] = test_results['Actual'].map({0: 'Did Not Survive', 1: 'Survived'})
    test_results['Predicted_Label'] = test_results['Predicted'].map({0: 'Did Not Survive', 1: 'Survived'})

    print("\n--- First 10 Test Predictions ---")
    print(test_results.head(10).to_string(index=False))

    print("\n--- Last 10 Test Predictions ---")
    print(test_results.tail(10).to_string(index=False))

    avg_survival_prob = np.mean(model.predict_proba(X)[:, 1]) * 100
    print(f"\nEstimated Average Survival Probability (All Data): {avg_survival_prob:.2f}%")

    print("\n=== Titanic Survival Prediction Complete ===")

if __name__ == "__main__":
    predict_titanic_survival()
