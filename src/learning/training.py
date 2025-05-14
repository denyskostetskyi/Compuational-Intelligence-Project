import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import constants
from .model_report import ModelReport

def train_rfc(df: pd.DataFrame) -> ModelReport:
    X_train_resampled, X_test, y_train_resampled, y_test = __split_and_oversample(df)
    __save_splitted_dataset(X_train_resampled, X_test, y_train_resampled, y_test)
    model = __train_random_forest_classifier(X_train_resampled, y_train_resampled)
    __save_model(model)
    report = __evaluate_model(model, X_test, y_test)
    return report

def __split_and_oversample(df: pd.DataFrame):
    y = df[constants.COLUMN_TARGET]
    X = df.drop(columns=[constants.COLUMN_TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=constants.TEST_DATASET_SIZE,
        stratify=y,
        random_state=constants.RANDOM_STATE
    )

    print("Before oversampling:", Counter(y_train))
    smote = SMOTE(random_state=constants.RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("After oversampling:", Counter(y_train_resampled))
    return X_train_resampled, X_test, y_train_resampled, y_test

def __save_splitted_dataset(X_train, X_test, y_train, y_test):
    X_train.to_csv(constants.PATH_X_TRAIN, index=False)
    X_test.to_csv(constants.PATH_X_TEST, index=False)
    y_train.to_csv(constants.PATH_Y_TRAIN, index=False)
    y_test.to_csv(constants.PATH_Y_TEST, index=False)
    print("Datasets saved successfully.")

def __train_random_forest_classifier(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        bootstrap=False,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=constants.RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("Random Forest Classifier trained successfully.")
    return model

def __save_model(model: RandomForestClassifier) -> None:
    joblib.dump(model, constants.PATH_MODEL_RANDOM_FOREST_CLASSIFIER)
    print(f"Trained model saved to {constants.PATH_MODEL_RANDOM_FOREST_CLASSIFIER}.")

def __evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> ModelReport:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    classification_rep = classification_report(
        y_test, 
        y_pred, 
        target_names=list(constants.TARGET_MAPPING.keys())
    )
    print("Classification Report:\n", classification_rep)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Calculate macro-averaged ROC AUC
    classes = np.unique(y_test)
    y_true_bin = label_binarize(y_test, classes=classes)

    fpr = dict()
    tpr = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)
    print(f"Macro-Averaged ROC AUC: {macro_auc:.4f}")

    return ModelReport.from_raw(
        model_name=type(model).__name__,
        target_classes_mapping=constants.TARGET_MAPPING,
        accuracy=accuracy,
        confusion_matrix=cm,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=macro_auc
    )
