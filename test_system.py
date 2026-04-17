#!/usr/bin/env python3
"""
CNC RAG System Test Script

Sistemin temel fonksiyonlarýný test eder.
"""

import os
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ingest import ingest_data
from core.retriever import create_rag_system


def test_data_ingestion():
    """Veri yükleme i÷lemini test eder."""
    print("=== Veri Yükleme Testi ===")
    
    # Test PDF dosyasý oluþtur (varsa)
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print("data/ dizini oluþturuldu.")
    
    # data/ dizininde PDF varsa test et
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("Uyarý: data/ dizininde PDF dosyasý bulunamadý.")
        print("Test için lütfen PDF dosyalarýný data/ klasörüne koyun.")
        return False
    
    try:
        vector_store = ingest_data()
        if vector_store:
            print("â Veri yükleme baþarýlý!")
            return True
        else:
            print("â Veri yükleme baþarýsýz!")
            return False
    except Exception as e:
        print(f"â Veri yükleme hatasý: {str(e)}")
        return False


def test_rag_system():
    """RAG sistemini test eder."""
    print("\n=== RAG Sistemi Testi ===")
    
    try:
        rag = create_rag_system()
        
        # Sistem bilgisini al
        info = rag.get_system_info()
        print("Sistem Bilgileri:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test sorusu
        test_question = "Sistem test sorusu"
        print(f"\nTest sorusu: {test_question}")
        
        result = rag.ask_question(test_question)
        print(f"Cevap: {result['answer']}")
        
        if result['source_documents']:
            print("Kaynak dokümanlar bulundu.")
        else:
            print("Kaynak doküman bulunamadý (normal, test sorusu için).")
        
        print("â RAG sistemi testi baþarýlý!")
        return True
        
    except Exception as e:
        print(f"â RAG sistemi hatasý: {str(e)}")
        return False


def test_api_import():
    """API modülünün import edilmesini test eder."""
    print("\n=== API Import Testi ===")
    
    try:
        from api.main import app
        print("â API modülü baþarýyla import edildi!")
        
        # FastAPI app objesini kontrol et
        if hasattr(app, 'title'):
            print(f"  App Title: {app.title}")
        
        return True
        
    except Exception as e:
        print(f"â API import hatasý: {str(e)}")
        return False


def check_ollama_models():
    """Ollama modellerinin mevcut olup olmadýðýný kontrol eder."""
    print("\n=== Ollama Model Kontrolü ===")
    
    try:
        import requests
        
        # Ollama sunucusunu kontrol et
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            print(f"Mevcut modeller: {', '.join(model_names)}")
            
            required_models = ['llama3', 'nomic-embed-text']
            missing_models = [model for model in required_models if model not in model_names]
            
            if missing_models:
                print(f"Eksik modeller: {', '.join(missing_models)}")
                print(f"Çözüm: ollama pull {' '.join(missing_models)}")
                return False
            else:
                print("â Gerekli modeller mevcut!")
                return True
        else:
            print("â Ollama sunucusuna ulaþýlamadý!")
            print("Çözüm: 'ollama serve' komutunu çalýþtýrýn")
            return False
            
    except Exception as e:
        print(f"â Ollama kontrol hatasý: {str(e)}")
        print("Çözüm: Ollama'yý indirin ve kurun: https://ollama.ai/download")
        return False


def main():
    """Ana test fonksiyonu."""
    print("CNC RAG System Test\n")
    
    tests = [
        ("Ollama Model Kontrolü", check_ollama_models),
        ("API Import Testi", test_api_import),
        ("Veri Yükleme Testi", test_data_ingestion),
        ("RAG Sistemi Testi", test_rag_system),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"â {test_name} kritik hatasý: {str(e)}")
            results.append((test_name, False))
    
    # Sonuç özeti
    print("\n" + "="*50)
    print("TEST ÖZETÄ°")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "â BAÅARILI" if result else "â BAÅARISIZ"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nToplam: {passed}/{total} test baþarýlý")
    
    if passed == total:
        print("ð Tüm testler baþarýlý! Sistem kullanýma hazýr.")
    else:
        print("â Bazý testler baþarýsýz. Lütfen yukarýdaki hatalarý düzeltin.")
    
    print("\nBaþlangýç için:")
    print("1. PDF dosyalarýný data/ klasörüne koyun")
    print("2. 'python -m core.ingest' çalýþtýrýn")
    print("3. 'python -m api.main' ile sunucuyu baþlatýn")
    print("4. http://localhost:8000 adresini ziyaret edin")


if __name__ == "__main__":
    main()
