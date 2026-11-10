"""
Otel yorumları analizi  - GRU + Decision Tree
Veri seti negative ve positive yorumlar içerir. İkili sınıflandırma problemi çözmek.
Veri seti açıklaması = https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
-otel adresi , review date , hotel name , reviewer nationality,
-negative review: 0
-positive review: 1
-örnek:
     -negative review: room was dirty 
     -positive review: no positive
     --> negative
Proje Akış Şeması ( plan program):
1. Veri Yükleme 
2. Metin Seçme( negative review/ positive review)
3.etiket oluşturma
4.Metin Temizleme
5.Train/test split
6.Decision Tree Modeli Oluşturma 
7.GRU modeli eğitme 
8.Değerlendirme ( Accuracy , F1, Confusion Matrix)
9.Model Kaydetme


Gerekli Kütüphaneler:
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn nltk
"""
#gerekli kütüphanelerin içeriye aktarılması 
import pandas as pd 
import numpy as np 
import re
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU , Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#veri yükleme
#metin seçme
#etiket oluşturma
df = pd.read_csv("Hotel_Reviews.csv")
print(df.head())
print(df.columns)

#etiketleme kuralı 
def extract_review_and_label(row):
    neg = row["Negative_Review"].strip()
    pos = row["Positive_Review"].strip()

    #negative yorum varsa
    if neg != "No Negative":
          return neg, 0  # negative review, label 0
    if pos != "No Positive":
          return pos, 1  # positive review, label 1
    return pos, 1


#yeni kolonları oluştur
df["review"], df["label"] = zip(*df.apply(extract_review_and_label, axis=1))
print(df.head()) 

#4. metin temizleme
def clean_text(text):

     text = text.lower()
     text = re.sub(r"https\S+", "", text)  # remove URLs
     text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)  # remove punctuation and numbers
     text = re.sub(r"\s+", " ", text)  # remove extra whitespace
     return text

#print(clean_text(" Merhaba 12345 # / <div>"))  # test the function
df["clean_review"] = df["review"].apply(clean_text)
print(df.head(10))
# 5. train test split
X = df["clean_review"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")



#6. decision tree model training : metin temsili + dt
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

dt = DecisionTreeClassifier(max_depth=13)
dt.fit(X_train_tfidf, y_train)
pred_dt = dt.predict(X_test_tfidf)

print("Decision Tree")
print(f"Accuracy: {accuracy_score(y_test, pred_dt)}")
print(f"F1 Score: {f1_score(y_test, pred_dt)}")

"""
Decision Tree 
     max_depth=13
     Accuracy: 0.905
     F1 Score: 0.905

"""

#7.GRU modeli eğit
#8.Değerlendirme ( Accuracy , F1, Confusion Matrix
#9 model kaydetme

# gru model 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = max(len(seq) for seq in X_train_seq)
#en uzun cümleyi bul

# tüm cümleleri aynı uzunluğa getirmek için padding işlemi uygula
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

vocab_size = len(tokenizer.word_index) + 1 

model = Sequential(
      [

           Embedding(vocab_size, 128, input_length = max_len),
           GRU(128), # GRU BLOK
           Dropout(0.3), # overfittingi engellemek için
           Dense(1, activation='sigmoid') #ikili sınıflandırma için
      ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
     X_train_pad, #feature ( kelimeler)
     y_train, #etiketler
     epochs = 1, #training sayısı
     batch_size = 256, # 1 kerede eğitilecek veri sayısı 
     validation_split = 0.1, # validation için ayırma oranı
     verbose = 1 # eğitim süreci hakkında bilgi ver
)
pred_gru = (model.predict(X_test_pad) > 0.5).astype(int)

print("gru:")
print(f"Accuracy: {accuracy_score(y_test, pred_gru)}")
print(f"F1: {f1_score(y_test, pred_gru)}")

cm = confusion_matrix(y_test, pred_gru)
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens")
plt.title("Confusion Matrix - GRU")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#model kaydetme
model.save("model_sentiment.h5")

with open("tokenizer_sentiment.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


print("Eğitim tamamlandı")







