from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def run_classical_baseline(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n--- Classical Baseline (XGBoost) Report ---")
    print(classification_report(y_test, preds))
    return model