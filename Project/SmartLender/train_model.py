import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(path=os.path.join('data','loan_prediction.csv')):
    df = pd.read_csv(path)
    return df


def build_and_select_model(df, save_path=os.path.join(BASE_DIR, 'models', 'loan_model.pkl')):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = df.copy()
    # show class distribution
    print('Class distribution:', df['Loan_Status'].value_counts().to_dict())
    y = df['Loan_Status'].map({'Y':1, 'N':0})
    X = df.drop(columns=['Loan_Status'])

    numeric_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    models = {
        'logistic': LogisticRegression(max_iter=2000),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'knn': KNeighborsClassifier()
    }
    if _HAS_XGB:
        models['xgboost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    best_score = -1
    best_name = None
    best_pipeline = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        try:
            pipe.fit(X_train, y_train)
            if hasattr(pipe, 'predict_proba') or hasattr(pipe.named_steps['classifier'], 'predict_proba'):
                probs = pipe.predict_proba(X_test)[:,1]
                score = roc_auc_score(y_test, probs)
            else:
                preds = pipe.predict(X_test)
                score = accuracy_score(y_test, preds)
            print(f'{name} score: {score:.4f}')
            if score > best_score:
                best_score = score
                best_name = name
                best_pipeline = pipe
        except Exception as e:
            print('Failed training', name, e)

    if best_pipeline is None:
        raise RuntimeError('No model could be trained')

    # evaluate selected model on test
    preds = best_pipeline.predict(X_test)
    probs = best_pipeline.predict_proba(X_test)[:,1] if hasattr(best_pipeline, 'predict_proba') or hasattr(best_pipeline.named_steps['classifier'], 'predict_proba') else None
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if probs is not None else None
    print('Best model:', best_name, 'accuracy:', acc, 'auc:', auc)

    # save with joblib for better compatibility
    joblib.dump(best_pipeline, save_path)
    print('Saved best model to', save_path)

    # also save metadata
    meta = {'model': best_name, 'accuracy': float(acc), 'auc': float(auc) if auc is not None else None}
    with open(os.path.join(os.path.dirname(save_path), 'model_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return best_pipeline, meta


if __name__ == '__main__':
    df = load_data()
    build_and_select_model(df)
