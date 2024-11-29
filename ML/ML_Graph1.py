import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


# Function to read AST files and return a list of texts
def read_ast_files(directory):
    ast_texts = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            ast_texts.append(file.read())
    return ast_texts


# Function to gather data from directories with a specific label
def gather_data_from_directories(directories, label):
    all_texts = []
    all_labels = []
    for directory in directories:
        ast_texts = read_ast_files(directory)
        all_texts.extend(ast_texts)
        all_labels.extend([label] * len(ast_texts))
    return all_texts, all_labels


# Data paths
benign_ast_dirs = [
    'D:/PowerShell/Extraction/GithubGist/GithubGist(ast)',
    'D:/PowerShell/Extraction/Technet/Technet(ast)',
    'D:/PowerShell/Extraction/PoshCode/PoshCode(ast)'
]

obfuscated_benign_ast_dirs = [
    'D:/PowerShell/Extraction/InvokeCradleCrafter/InvokeCradleCrafter(ast)',
    'D:/PowerShell/Extraction/InvokeObfuscation/InvokeObfuscation(ast)',
    'D:/PowerShell/Extraction/IseSteroids/IseSteroids(ast)'
]

malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/decoded/decoded(ast)'
obfuscated_malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/encoded/mentor(ast)'

# Load data
benign_texts, benign_labels = gather_data_from_directories(benign_ast_dirs, label=0)
obfuscated_benign_texts, obfuscated_benign_labels = gather_data_from_directories(obfuscated_benign_ast_dirs, label=1)
malicious_texts, malicious_labels = gather_data_from_directories([malicious_ast_dir], label=2)
obfuscated_malicious_texts, obfuscated_malicious_labels = gather_data_from_directories([obfuscated_malicious_ast_dir],
                                                                                       label=3)

# Combine all texts and labels
all_texts = benign_texts + obfuscated_benign_texts + malicious_texts + obfuscated_malicious_texts
all_labels = benign_labels + obfuscated_benign_labels + malicious_labels + obfuscated_malicious_labels
y = np.array(all_labels)

# Vectorize data for AST 3-gram and TF-IDF 3-gram
vectorizer_ast = CountVectorizer(ngram_range=(3, 3))
X_ast_count = vectorizer_ast.fit_transform(all_texts).toarray()

vectorizer_tfidf = TfidfVectorizer(ngram_range=(3, 3))
X_ast_tfidf = vectorizer_tfidf.fit_transform(all_texts).toarray()

# Simple AST Vectorization (No n-grams)
vectorizer_simple = CountVectorizer()
X_ast_simple = vectorizer_simple.fit_transform(all_texts).toarray()


# Function to train and add Precision-Recall Curve for a model for class 1 only
def add_precision_recall_curve_class1(X, y, model, label):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining {label}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Evaluate model
    print(f"\n{label} - Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Plot Precision-Recall Curve for class 1 only
    precision, recall, _ = precision_recall_curve(y_test == 1, y_proba[:, 1])
    avg_precision = average_precision_score(y_test == 1, y_proba[:, 1])
    plt.plot(recall, precision, label=f'{label} - Class 1 (AP={avg_precision:.2f})')


# Plot setup for all curves
plt.figure(figsize=(10, 8))

# Add Precision-Recall Curves for each model and vectorization method for class 1
add_precision_recall_curve_class1(X_ast_count, y, RandomForestClassifier(n_estimators=100, random_state=42),
                                  "AST 3-gram Random Forest")
add_precision_recall_curve_class1(X_ast_tfidf, y, RandomForestClassifier(n_estimators=100, random_state=42),
                                  "TF-IDF 3-gram Random Forest")
add_precision_recall_curve_class1(X_ast_simple, y, RandomForestClassifier(n_estimators=100, random_state=42),
                                  "AST Simple Random Forest")

# Uncomment to add the Gradient Boosting curve for class 1
# add_precision_recall_curve_class1(X_ast_simple, y, GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42), "AST Simple Gradient Boosting")

# Finalize the plot
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Combined Precision-Recall Curves for Class 1')
plt.legend(loc='best')
plt.show()
