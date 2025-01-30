from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

vectorizer = TfidfVectorizer(max_features=10000)
# Load the 20 newsgroups dataset already vectorized
newsgroups_train = fetch_20newsgroups_vectorized(subset='train')
newsgroups_test = fetch_20newsgroups_vectorized(subset='test')

# The data is already vectorized, so you can directly access it
X_train_tfidf = newsgroups_train.data
X_test_tfidf = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Feature selection using SelectKBest with chi2
k = 1000  # Number of top features to select, adjust as needed
selector = SelectKBest(score_func=chi2, k=k)
X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = selector.transform(X_test_tfidf)

# Train logistic regression
clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(X_train_selected, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_selected)
print(classification_report(y_test, y_pred))

# Print the shape of the vectorized data
print(f"Train data shape: {X_train_tfidf.shape}")
print(f"Test data shape: {X_test_tfidf.shape}")
