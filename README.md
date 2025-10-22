# 🥗 Yapay Zeka Beslenme Asistanı (Gemini RAG)
Bu proje, Retrieval-Augmented Generation (RAG) mimarisini kullanarak, 10.000'den fazla besin değerini içeren özel bir veri seti üzerinde doğru ve bağlamsal cevaplar üreten interaktif bir Streamlit sohbet uygulamasıdır. Kullanıcılar, besinlerin kalori, protein, yağ gibi değerlerini Türkçe olarak sorgulayabilir ve birden fazla besin arasında karşılaştırma yapabilirler.


# 🚀 Canlı Uygulama
(BURAYA Streamlit Cloud'a Deploy Ettikten Sonra Alacağınız Canlı Linki Ekleyeceksiniz)$$Canlı Uygulama Linki Buraya Gelecek$$



# ⚙️ Kullanılan Teknolojiler ve Mimarisi 
 LLM (Büyük Dil Modeli): Google Gemini 2.5 
 
FlashRAG Mimarisi: LangChain (Prompt Templates, Zincirleme)

 Vektör Veritabanı: FAISS (Büyük veri setinin hızlı aranabilir hale getirilmesi için)

 Embedding Modeli: GoogleGenerativeAIEmbeddings (text-embedding-004)

Web Arayüzü: Streamlit

 Veri Seti: daily_food_nutrition_dataset.csv (10.000+ kayıt)



# 🛠️ Kurulum ve Yerel Çalıştırma
Bu projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları takip edin.
1. Dosya Yapısı Kontrolü
   Deponun kök dizininde aşağıdaki dosyalar bulunmalıdır: project.py (Streamlit uygulaması),requirements.txt (Gerekli kütüphaneler),README.md ,LICENSE ,to-do.md ,.env 
2. Sanal Ortam Kurulumu
Proje bağımlılıklarını izole etmek için sanal ortam kurun ve requirements.txt dosyasındaki tüm kütüphaneleri yükleyin:
 Sanal ortam oluşturma (isteğe bağlı ama önerilir)
python -m venv venv 
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows
3. Kütüphaneleri yükleme
pip install -r requirements.txt

4. API Anahtarını Tanımlama
   Google Gemini API anahtarınızı içeren bir .env dosyası oluşturun ve anahtarınızı içine yazın  :  # .env dosyası GEMINI_API_KEY="SİZİN_ANAHTARINIZ" 
5. Veri Setini Ekleme
   Veri seti (daily_food_nutrition_dataset.csv), proje klasörüne manuel olarak yerleştirilmelidir.Veri Seti Kaynağı (Kaggle): Dosyanın orijinaline ve indirme sayfasına aşağıdaki linkten ulaşabilirsiniz. (https://www.kaggle.com/datasets/adilshamim8/daily-food-and-nutrition-dataset) daily_food_nutrition_dataset.csv dosyasını, tam olarak project.py dosyasının bulunduğu dizine yerleştirin.
6. Uygulamayı Başlatma
   Uygulamayı yerelde başlatmak için Streamlit komutunu kullanın :streamlit run project.py


  #  Veri Seti Hazırlama
Veri Seti Hakkında Bilgi
Projede kullanılan veri seti, Kaggle platformundan temin edilmiştir.

Adı: Daily Food Nutrition Dataset

İçerik: 10.000'den fazla farklı besin maddesinin porsiyon büyüklükleri, kalori, protein, yağ, karbonhidrat, lif ve şeker değerlerini içeren tablosal veridir.

Format: UTF-8 kodlamalı CSV dosyası (daily_food_nutrition_dataset.csv).



# ⚠️ Önemli Not:
Veri Seti Kısıtlaması: Proje adımları gereği, veri seti (daily_food_nutrition_dataset.csv) GitHub deposuna dahil edilmemiştir (.gitignore dosyasındadır). Bu nedenle, Streamlit Cloud'da yayınlanan canlı uygulamada FileNotFoundError hatası görülecektir.

Türkçe Arama Sınırlaması: Veri setimizdeki besin isimleri İngilizce olarak indekslenmiştir (Egg, Banana). RAG mimarisinin veri akışı gereği, Retriever (Arayıcı) Türkçe bir kelimeyi (örn: "Yumurta") doğrudan İngilizce indekslenmiş veritabanında arar. Bu arama çoğunlukla başarısız olduğu için, modelin eline boş Bağlam (Context) gider ve "bilgi yok" yanıtı alınır.
Bu, RAG'ın veri diline olan bağlılığından kaynaklanan mimari bir kısıtlamadır.
Test için, veri setinde bulunan İngilizce karşılıkların (örn: "Egg", "Banana") kullanılması gerekmektedir.

# Çalışma Akışı ve Test Kabiliyetleri
Kullanıcı, arayüzdeki sohbet kutusuna besinlerle ilgili sorularını sorabilir.

Soru Örneği (Tek Değer): "Banana kalori değeri nedir?"

Soru Örneği (Karşılaştırma): "Apple mı yoksa banana mu şeker daha yüksek?"


  ### Web Arayüzü Görüntüsü
![Uygulama Arayüzü](images/uygulama_arayuzu.png)

### Karşılaştırma Yeteneği Testi
![Karşılaştırma Testi](images/karşilaştirma_testi.png)
### Tek Değer Bulma Testi 
![Tek Değer Testi Testi](images/tek_deger_testi.png)
