"""
Otel yorumları duygu analizi projesi için streamlit dashboard
 bu dosyada , GRU tabanlı sentiment analiz apısini kullanrak etkileşimli bir kullancı arayüzü oluşturur
 arayüz iki ana sütundan oluşacak.


 sol sütun ( user input panel)
  - kullanıcı bir otel yorumu yapar.
  -tahmin et butonuna basar.
  -UI ( stramlit ) , Fastapida ki /predict_sentiment endpoint ine POST isteği atar.
  -apiden gelen cevapları sonuç ekranında gösterir.


  sağ sütun ( otel yorumları ve sonuç ekranı)
   -pozitif yorum sayısı
   -negatif yorum sayısı
   -kullanıcı her analiz yaptığında sayaçlar artar

bu tasarım gerçek bir kullanıcı arayüzü için temel bir örnek teşkil eder. Kullanıcı deneyimini geliştirmek için ek özellikler ve görselleştirmeler eklenebilir.

API bağlantısı :
POST /predict_sentiment
Body: { "review": "kullanıcı yorumu" }
Response: { "sentiment": "positive" } veya { "sentiment": "negative



nasıl çalıştırılır:
 - api.py dosyasını çalıştırın (uvicorn api:app --reload)
    - streamlit dashboardu çalıştırın (streamlit run 3_streamlit.py)
    -tarayıcıda açılan sayfada kullanıcı arayüzü ile etkileşime geçin.

Gerekli kurulumlar:

pip install streamlit requests


"""
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict_sentiment"

st.set_page_config(page_title="Otel Yorumları Duygu Analizi", layout="wide")
st.title("Otel Yorumları Duygu Analizi Dashboard")
st.write("GRU modelini kullanan gerçek zamanlı bir duygu analizi uygulaması")

# Sayaçlar
if "positive_count" not in st.session_state:
    st.session_state["positive_count"] = 0
if "negative_count" not in st.session_state:
    st.session_state["negative_count"] = 0

col1, col2 = st.columns(2)

# Sol sütun: yorum girişi
with col1:
    st.subheader("Otel yorumunuzu girin")

    user_input = st.text_area(
        "Yorum",
        placeholder="Otel yorumunuzu buraya yazın...",
        height=250,
    )

    if st.button("Tahmin Et"):
        if not user_input.strip():
            st.warning("Lütfen bir yorum girin.")
        else:
            try:
                response = requests.post(
                    API_URL, json={"review": user_input}, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    label = result["label"].strip().lower()
                    prob = result["probability"]

                    if label in ("pozitif", "positive"):
                        st.session_state["positive_count"] += 1
                        st.success(f"Yorum POZİTİF olarak sınıflandırıldı. (Olasılık: {prob:.2f})")
                    else:
                        st.session_state["negative_count"] += 1
                        st.error(f"Yorum NEGATİF olarak sınıflandırıldı. (Olasılık: {prob:.2f})")
                else:
                    st.error(f"API hatası: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("API'ye bağlanılamadı. uvicorn'un çalıştığından emin ol.")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")


#sağ sütun dashboard paneli 
with col2:
    st.subheader("otel duygu analizi paneli")

    
    st.metric("Pozitif yorum sayısı", st.session_state["positive_count"])
    st.metric("Negatif yorum sayısı", st.session_state["negative_count"])
    st.write("------")
    st.caption("bu sayaçlar sadece bu oturum boyunca geçerlidir")





# Sağ sütun: sayaçlar

  