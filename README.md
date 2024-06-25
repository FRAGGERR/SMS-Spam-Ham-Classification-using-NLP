---

# SMS Spam Detection

This repository contains code for detecting spam messages using natural language processing and machine learning techniques. The dataset used is the SMS Spam Collection Data Set, which contains a collection of SMS messages tagged as spam or ham (non-spam).

## Project Structure

- `smsspamcollection/SMSSpamCollection`: The dataset file containing SMS messages and their corresponding labels.
- `spam_detection.py`: The main Python script containing the code for preprocessing, training, and evaluating the spam detection models.

## Requirements

The following Python libraries are required to run the code:

- pandas
- numpy
- scikit-learn
- nltk

You can install the required libraries using pip:

```sh
pip install pandas numpy scikit-learn nltk
```

## Usage

### Dataset

The dataset should be placed in the `smsspamcollection` directory with the name `SMSSpamCollection`. The file should be tab-separated and contain two columns: `label` and `message`.

### Preprocessing

The code preprocesses the messages by:

1. Removing non-alphabetical characters.
2. Converting text to lowercase.
3. Tokenizing the text.
4. Removing stopwords.
5. Applying stemming using the Porter Stemmer.

### Feature Extraction

Two types of feature extraction methods are used:

1. **Bag of Words (BOW)**: Using `CountVectorizer` to create a matrix of token counts.
2. **TF-IDF**: Using `TfidfVectorizer` to create a matrix of TF-IDF features.

### Model Training and Evaluation

Two models are trained and evaluated using the Naive Bayes classifier:

1. **BOW Model**:
    - The model is trained using the training data and BOW features.
    - The performance is evaluated using accuracy and classification report metrics.
    
2. **TF-IDF Model**:
    - The model is trained using the training data and TF-IDF features.
    - The performance is evaluated using accuracy and classification report metrics.

### Running the Script

You can run the script as follows:

```sh
python spam_detection.py
```

### Output

The script will output the accuracy and classification report for both the BOW and TF-IDF models.

## Code Explanation

Below is a brief explanation of the main sections of the code:

1. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report
    ```

2. **Load Dataset**:
    ```python
    messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])
    ```

3. **Preprocess Data**:
    ```python
    nltk.download('stopwords')
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    ```

4. **Prepare Labels**:
    ```python
    y = pd.get_dummies(messages['label'])
    y = y.iloc[:, 0].values
    ```

5. **Train Test Split**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.20)
    ```

6. **BOW Model**:
    ```python
    cv = CountVectorizer(max_features=2500, ngram_range=(1, 2))
    X_train = cv.fit_transform(X_train).toarray()
    X_test = cv.transform(X_test).toarray()
    spam_detect_model = MultinomialNB().fit(X_train, y_train)
    y_pred = spam_detect_model.predict(X_test)
    print("BOW Model Accuracy:", accuracy_score(y_test, y_pred))
    print("BOW Model Classification Report:\n", classification_report(y_test, y_pred))
    ```

7. **TF-IDF Model**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.20)
    tv = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X_train = tv.fit_transform(X_train).toarray()
    X_test = tv.transform(X_test).toarray()
    spam_tfidf_model = MultinomialNB().fit(X_train, y_train)
    y_pred = spam_tfidf_model.predict(X_test)
    print("TF-IDF Model Accuracy:", accuracy_score(y_test, y_pred))
    print("TF-IDF Model Classification Report:\n", classification_report(y_pred, y_test))
    ```

## Conclusion

This project demonstrates the use of natural language processing and machine learning techniques for spam detection in SMS messages. By comparing different feature extraction methods (BOW and TF-IDF), we can evaluate their effectiveness in classifying spam and ham messages.

Feel free to explore the code and experiment with different models and preprocessing techniques to improve the performance.


---
