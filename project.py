import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# API anahtarını .env dosyasından yükler. 
# Bu satır, Streamlit Cloud'da 'Secrets' mekanizmasını kullanacağımız için 
# yerel çalıştırmada gereklidir.
load_dotenv() 

# RAG Bileşenleri için Gerekli Kütüphaneler
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# API Anahtarının Tanımlı Olduğunu Kontrol Etme (Güvenlik Kontrolü)
if "GEMINI_API_KEY" not in os.environ:
    st.error("Hata: GEMINI_API_KEY bulunamadı. Lütfen '.env' dosyasını oluşturun veya Streamlit Secrets'a ekleyin.")
    # st.stop() # Eğer eksikse uygulamanın çalışmasını durdurabiliriz


# @st.cache_resource: Bu fonksiyonun çıktısı (docs_chunks) bir daha çalıştırılmaz. 
# Böylece uygulama her yenilendiğinde CSV tekrar yüklenmez.
@st.cache_resource
def load_and_chunk_data():
    """CSV dosyasını yükler, parçalara ayırır ve LangChain dokümanlarını döndürür."""
    
    # 🚨 NOT: Dosya yolunuzun doğru olduğundan emin olun! (Proje kök dizininde olmalı)
    csv_path = "daily_food_nutrition_dataset.csv" 
    
    try:
        # 1. Veri Yükleme (CSVLoader)
        loader = CSVLoader(file_path=csv_path)
        documents = loader.load()
        
        # 2. Parçalara Ayırma (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        docs_chunks = text_splitter.split_documents(documents)
        
        # Konsola değil, Streamlit çıktısına yazdırıyoruz
        st.sidebar.success(f"Veri yükleme tamamlandı. {len(docs_chunks)} parça hazır.")
        return docs_chunks

    except Exception as e:
        st.error(f"HATA: Veri yüklenirken bir sorun oluştu. Dosya yolu doğru mu? Detay: {e}")
        return None

# Fonksiyonu çağırarak docs_chunks değişkenini uygulama genelinde kullanılabilir hale getiriyoruz
docs_chunks = load_and_chunk_data()
# --- 4. HÜCRE: RAG ZİNCİRİNİ KURMA (Tek Seferlik Çalıştırma) ---

@st.cache_resource
def setup_rag_chain(docs_chunks):
    """Embedding modelini, Vektör Depolamayı ve LCEL zincirini kurar."""
    
    # 1. Gömme Modeli (Embedding)
    # API key, os.environ'dan otomatik olarak alınacaktır
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004", 
        google_api_key=os.environ["GEMINI_API_KEY"] 
    )

    # 2. Vektör Veritabanı (FAISS)
    try:
        vector_store = FAISS.from_documents(
            documents=docs_chunks, 
            embedding=embedding_model
        )
        st.sidebar.success("Vektör Veritabanı (FAISS) hazır.")
    except Exception as e:
        st.error(f"Vektör Veritabanı oluşturulurken hata: {e}")
        return None

    # 3. Arama (Retrieval) fonksiyonunu tanımlama
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # k=3'ten k=5'e çıkarıldı

    # 4. LLM Tanımı
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

  

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser() # OutputParser eklemek, temiz string cevap almanızı garanti eder.
    )
    st.sidebar.success("RAG Zinciri (LCEL) başarıyla kuruldu.") 

    return qa_chain
# prompt giriyoruz 
template = """Sen, besin değerleri konusunda uzmanlaşmış bir AI Beslenme Asistanısın. Görevin, TÜM Türkçe sorguları, Bağlam'daki İngilizce verilere göre cevaplamaktır. Cevap, sadece bu Bağlam'a dayanmak zorundadır.

KESİNLİKLE UYMAN GEREKEN ZİNCİRLEME VE SÜZME TALİMATI:

1. **ÖNCELİKLİ ÇEVİRİ ZORUNLULUĞU (CRITICAL):** Kullanıcının Türkçe sorusundaki besin adını (örn: 'muz' -> 'Banana', 'yumurta' -> 'Egg') bul. Bu İngilizce karşılığı, Bağlam'daki verileri analiz etmek için temel alacağın anahtar kelime olacaktır.

2. **BAĞLAMSAL SÜZME VE TEK DEĞER KISITLAMASI:** Arama sonucu Bağlam'da birden fazla kayıt gelirse, çevirdiğin İngilizce anahtar kelimeye uygun olarak (Örn: 'Egg'), Bağlam'dan **yalnızca EN YAYGIN TEK BİR DEĞERİ** çek. **KESİNLİKLE aralık (ila/arası) belirterek cevap verme.**

3. **HESAPLAMA & KARŞILAŞTIRMA:** Karşılaştırma veya matematiksel analizler istenirse, Bağlam'daki değerleri kullanarak bu hesaplamaları yap ve sonucu net bir şekilde ifade et.

4. **ÇIKTI FORMATI:** Cevabını kısa, net, profesyonel BİR Türkçe cümle olarak ver. Yanıtında asla İngilizce kelime kullanma.

5. **BİLGİ YOKSA (SADECE KESİNLİKLE BULUNAMAZSA):** Eğer çeviri ve arama sonucunda kesinlikle bir bilgi bulamazsan, sadece şu cümleyi kullan: 'Bu bilgi veri setinde bulunmamaktadır.'

Bağlam:
{context}

Kullanıcının Türkçe Sorusu: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Eğer veri yükleme başarılıysa zinciri kur
if docs_chunks:
    qa_chain = setup_rag_chain(docs_chunks)
else:
    qa_chain = None
    # --- NOTEBOOK 5. HÜCRESİNİN MANTIĞI: STREAMLIT WEB ARAYÜZÜNE TAŞINDI ---

# Uygulama başlığını ve genel açıklamasını ayarla
st.title("🍔 Gelişmiş Beslenme RAG Asistanı")
st.markdown("Verilen CSV verisine dayanarak besin değerleri, kategoriler ve karşılaştırmalar hakkında sorular sorun.")

# 1. QA Zincirinin Kullanılabilirliğini Kontrol Etme
if qa_chain is None:
    st.warning("RAG Asistanı şu an kullanılamıyor. Lütfen hata mesajlarını kontrol edin.")
    st.stop() # Uygulamanın daha fazla çalışmasını durdur

# 2. Kullanıcı Giriş Alanını Oluşturma
user_query = st.text_input(
    "Sorunuzu Buraya Yazın:", 
    placeholder="Örn: Banana kalori, protein ve şeker değerleri nedir?"
)

# 3. Sorguyu Çalıştırma (ask_chatbot_final fonksiyonunun yaptığı iş)
if user_query:
    st.info(f"Soru: {user_query}")
    
    with st.spinner("🔍 Cevap, Beslenme Veri Setinden Çekiliyor..."):
        try:
            # RAG Zincirini çağır
            response = qa_chain.invoke(user_query)
            
            # Cevabı Streamlit'te gösterme
            st.success("🤖 Asistan Cevabı:")
            st.markdown(response)
            
        except Exception as e:
            st.error(f"HATA: Soru yanıtlanırken bir sorun oluştu. Detay: {e}")