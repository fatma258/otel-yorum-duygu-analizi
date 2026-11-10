"""
API özeti:
Bu servisin görevi:
    -kullanıcıdan bir otel yorumu almak 
    -yorum metnini temizlemek
    -tokenizer ile sayısal dizilere dönüştürmek
    -GRU modeli ile pozitif ya da negatif tahmini yapmak
    -JSON formatında cevap return etmek
        -{
           "label": "pozitif",
           "probability": 0.91
                
        }
ENDPOINT (POST)

URL:
    - /predict_sentiment

İstek gövdesi:
    {
     
        "review": "Otel çok güzeldi, personel çok ilgiliydi."
    }

Cevap gövdesi:
    {
        "label": "pozitif",
        "probability": 0.98
    }
ÇALIŞTIRMA
    1.model dosyalarının bulunduğundan emin ol.
         -model_sentiment.h5
         -tokenizer_sentiment.pkl
    2.API yı başlat:
        uvicorn 2_api:app --reload
    3.tarayıcıda test et (swagger)
        http://127.0.0.1:8000/docs

        
GEREKLİ KÜTÜPHANELER
yalnızca API için gereken
pip install fastapi uvicorn

"""
#gerekli kütüphaneleri içe aktar 
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn

#fast api uygulaması 
app = FastAPI(
    title = "Otel Yorumları Sentiment Analizi",
    description = "farklı veri yapısı ile duygu analizi gerçekleştiren api",
    version = "1.0.0"
)
#model ve tokenizer dosyalarını yükle
#gru modeli yükle 
model = tf.keras.models.load_model("model_sentiment.h5")

with open("tokenizer_sentiment.pkl", "rb") as f:
    tokenizer = pickle.load(f)

#model input uzunluğu 
max_len = model.input_shape[1]

#metin temizleme
def clean_text(text: str) -> str:
    text = text.lower()  # küçük harfe çevir
    text = re.sub(r"http\S+", "", text)  # URL'leri kaldır
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # özel karakterleri kaldır
    text = re.sub(r"\s+", " ", text).strip()  # fazla boşlukları kaldır
    return text

#request (istek) / response (cevap) veri yapısı
class ReviewRequest(BaseModel):  #input
    review: str

class ReviewResponse(BaseModel):  #output
    label: str #positive/negative
    probability: float #olasılık 

#endpoint
@app.post("/predict_sentiment", response_model=ReviewResponse)
def predict_sentiment(req: ReviewRequest):
    """
    kullanıcıdan gelen otel yorumumun pozitif mi yoksa negatif mi olduğunu GRU modelini kullanark tahmin eder
    ADımlar : 
    1.yorum alır.
    2.clean_text ile yorumlar temziler
    3.tokenizer ile sayısal dizilere dönüştürür
    4. padding pad_sequences ile dizileri model input uzunluğuna getirir
    5.model  tahmin yapar
    6.probability >= 0.5 ise pozitif, < 0.5 ise negatif olarak label döndürür
    

    """
#   1. yorum metni 
    user_review = req.review
    #2.temizleme 
    cleaned = clean_text(user_review)
    #3. sayısal dizilere dönüştürme
    seq = tokenizer.texts_to_sequences([cleaned])
    #4. padding
    pad = pad_sequences(seq, maxlen=max_len)
    #5. tahmin
    prob = float(model.predict(pad)[0][0])
    #6. label belirleme
    label = "positive " if prob >= 0.5 else "negative"

    return ReviewResponse(label = label, probability = prob)

#lokalde çalıştırma 

if __name__ == "__main__":
   uvicorn.run("2_api:app", host = "127.0.0.1", port=8000, reload=True )    
