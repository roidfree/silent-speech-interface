"""Model factory returning a scikit-learn estimator pipeline."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def create_classifier(random_state: int = 42) -> Pipeline:
    """Return a basic classifier pipeline: scaler + random forest."""
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    return clf
