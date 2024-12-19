import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from razdel import tokenize
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

nltk.download('stopwords')

# Приклад датасету
data = {
    'text': [
        'Цей фільм просто чудовий! Мені дуже сподобалось.',
        'Я розчарований у цій книзі, вона нудна.',
        'Чудовий день для прогулянки!',
        'Мене дратує, коли мене ігнорують.',
        'Це було неймовірно весело!'
    ],
    'emotion': ['positive', 'negative', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)
print("Приклад даних:\n", df.head())

# Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)



stop_words = stopwords.words('ukrainian')

# Функція для очищення тексту
def preprocess_text(text):
    text = text.lower()  # до нижнього регістру
    text = re.sub(r'\W', ' ', text)  # видалення символів
    text = ' '.join([word.text for word in tokenize(text) if word.text not in stop_words])  # токенізація та стоп-слова
    return text

# Застосування очищення
X_train_clean = X_train.apply(preprocess_text)
X_test_clean = X_test.apply(preprocess_text)

# Векторизація TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_clean)
X_test_tfidf = vectorizer.transform(X_test_clean)

# Створення та навчання моделі
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Прогнозування
y_pred = model.predict(X_test_tfidf)

# Оцінка точності
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def analyze_emotion(text):
    clean_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([clean_text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

# Приклади аналізу
examples = [
    "Мені дуже сподобалася ця подорож!",
    "Це був жахливий день, я розлючений.",
    "Ваша допомога була неймовірно корисною, дякую!"
]

for example in examples:
    print(f"Текст: {example} --> Емоція: {analyze_emotion(example)}")



# Матриця помилок
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()