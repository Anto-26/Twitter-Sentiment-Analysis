import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y):
    """
    Train and evaluate Logistic Regression model.
    """

    # 1. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 2. Model
    model = LogisticRegression(max_iter=1000)

    # 3. Train
    model.fit(X_train, y_train)

    # 4. Predict
    y_pred = model.predict(X_test)

    # 5. Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return model