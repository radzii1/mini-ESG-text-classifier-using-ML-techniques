import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1) Load data
df = pd.read_csv("data/esg_samples.csv")

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3) Build pipeline: TF-IDF -> Linear SVM
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
    ("clf", LinearSVC())
])

# 4) Train
pipe.fit(X_train, y_train)

# 5) Evaluate
pred = pipe.predict(X_test)
print(classification_report(y_test, pred, digits=3))

# 6) Save model
joblib.dump(pipe, "esg_text_model.joblib")
print("Saved model -> esg_text_model.joblib")
