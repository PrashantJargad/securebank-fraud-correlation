"""
Peek inside the trained model (fraud_model.joblib).

    python inspect_model.py

The .joblib file is a saved model object, not data — you can't open it in a text
editor. This prints what it actually contains: its settings, how much weight it
gives each signal, and a look at one of its decision trees.
"""

from sklearn.tree import export_text

from ml import load_model, MODEL_PATH
from features import FEATURES


def main():
    model = load_model()

    print(f"Model file : {MODEL_PATH}")
    print(f"Type       : {type(model).__name__}")
    print(f"Trees      : {model.n_estimators_} decision trees")
    print(f"Max depth  : {model.max_depth}   Learning rate: {model.learning_rate}")

    print("\nHow much each signal drives the model's decisions:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_),
                            key=lambda t: -t[1]):
        bar = "#" * int(round(imp * 40))
        print(f"  {feat:20s} {imp:5.3f}  {bar}")

    print("\nOne of the model's decision trees (tree #1 of many), in plain rules:")
    first_tree = model.estimators_[0, 0]
    text = export_text(first_tree, feature_names=list(FEATURES), max_depth=3)
    print(text)


if __name__ == "__main__":
    main()
