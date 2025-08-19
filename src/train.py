# src/train.py
import pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

RANDOM_STATE = 42

# 1) Load data
df = pd.read_csv("data/esg_samples.csv")
X = df["text"].astype(str)
y = df["label"].astype(str)

# 2) Hold-out test split (stratified)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 3) Pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LinearSVC())
])

# 4) Hyperparameter search (proper ML way)
param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__max_features": [3000, 6000, 10000],
    "tfidf__max_df": [0.9, 0.95, 1.0],
    "tfidf__sublinear_tf": [True],
    "clf__C": [0.5, 1.0, 2.0],
    "clf__class_weight": [None, "balanced"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)

# 5) Fit on train folds; pick best by macro-F1
grid.fit(X_tr, y_tr)
print(f"Best CV Macro-F1: {grid.best_score_:.3f}")
print("Best params:", grid.best_params_)

# 6) Evaluate on the held-out test set
y_pred = grid.best_estimator_.predict(X_te)
print("\nHold-out classification report:\n")
print(classification_report(y_te, y_pred, digits=3))

# 7) Refit best pipeline on ALL data for deployment + save
best = grid.best_estimator_
best.fit(X, y)
joblib.dump(best, "esg_text_model.joblib")
print("Saved final model → esg_text_model.joblib")
