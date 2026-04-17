import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    """
    PDF dosyalarını belirtilen dizinden yükler.
    
    Args:
        directory_path: PDF dosyalarının bulunduğu dizin yolu
        
    Returns:
        List[Document]: Yüklenen dokümanlar listesi
    """
    documents = []
    pdf_directory = Path(directory_path)
    
    if not pdf_directory.exists():
        print(f"Dizin bulunamadı: {directory_path}")
        return documents
    
    # Dizindeki tüm PDF dosyalarını bul
    pdf_files = list(pdf_directory.glob("*.pdf"))
    
    if not pdf_files:
        print(f"{directory_path} dizininde PDF dosyası bulunamadı.")
        return documents
    
    print(f"Toplam {len(pdf_files)} PDF dosyası bulundu.")
    
    for pdf_file in pdf_files:
        try:
            print(f"Yükleniyor: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            pdf_docs = loader.load()
            
            # Her dokümana kaynak dosya bilgisini ekle
            for doc in pdf_docs:
                doc.metadata["source"] = pdf_file.name
                
            documents.extend(pdf_docs)
            print(f"+ {pdf_file.name} basariyla yuklendi ({len(pdf_docs)} sayfa)")
            
        except Exception as e:
            print(f"- {pdf_file.name} yüklenirken hata: {str(e)}")
    
    return documents


def create_vector_store(documents: List[Document], persist_directory: str = "chroma_db") -> Chroma:
    """
    Dokümanları vektörleştirir ve ChromaDB'ye kaydeder.
    
    Args:
        documents: Vektörleştirilecek dokümanlar
        persist_directory: Vektör veritabanının kaydedileceği dizin
        
    Returns:
        Chroma: Oluşturulan vektör deposu
    """
    if not documents:
        print("Vektörleştirilecek doküman bulunamadı!")
        return None
    
    print(f"\n{len(documents)} doküman vektörleştiriliyor...")
    
    # Text splitter oluştur
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        length_function=len,
    )
    
    # Dokümanları parçalara ayır
    chunks = text_splitter.split_documents(documents)
    print(f"Toplam {len(chunks)} parça oluşturuldu.")
    
    # Ollama embedding modeli
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # ChromaDB vektör deposu oluştur
    print("Vektör deposu oluşturuluyor...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Yeni Chroma versiyonlari otomatik kaydeder
    print(f"+ Vektör deposu {persist_directory} dizinine kaydedildi.")
    
    return vector_store


def ingest_data(data_directory: str = "data", persist_directory: str = "chroma_db_v2"):
    """
    Tam veri yükleme ve vektörleştirme sürecini çalıştırır.
    
    Args:
        data_directory: PDF dosyalarının bulunduğu dizin
        persist_directory: Vektör veritabanının kaydedileceği dizin
    """
    print("=== CNC RAG Sistemi Veri Yükleme Başlıyor ===\n")
    
    # PDF dosyalarını yükle
    documents = load_pdfs_from_directory(data_directory)
    
    if not documents:
        print("Hiç doküman yüklenemedi. İşlem sonlandırılıyor.")
        return None
    
    # Vektör deposu oluştur
    vector_store = create_vector_store(documents, persist_directory)
    
    if vector_store:
        print("\n=== Veri yükleme basariyla tamamlandi ===")
        return vector_store
    else:
        print("\n=== Veri yükleme basarisiz oldu ===")
        return None


if __name__ == "__main__":
    # Script doğrudan çalıştırıldığında veri yükleme sürecini başlat
    ingest_data()
