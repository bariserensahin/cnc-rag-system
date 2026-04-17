from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retriever import CNCRAGSystem


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]


class SystemInfo(BaseModel):
    system_status: str
    persist_directory: str
    embedding_model: str
    llm_model: str
    document_count: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]


# FastAPI uygulamasi
app = FastAPI(
    title="CNC RAG System API",
    description="Fabrika Bakim Asistani icin 100% Lokal RAG Sistemi",
    version="1.0.0"
)

# Global RAG sistemi ornegi
rag_system = None


def get_rag_system():
    """RAG sistemini lazy loading ile olusturur."""
    global rag_system
    if rag_system is None:
        try:
            rag_system = CNCRAGSystem()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"RAG sistemi yuklenemedi: {str(e)}. Once veri yukleme islemi yapin."
            )
    return rag_system


@app.get("/")
async def root():
    """API ana sayfasi."""
    return {
        "message": "CNC RAG System API",
        "description": "Fabrika Bakim Asistani icin 100% Lokal RAG Sistemi",
        "endpoints": {
            "ask": "/ask - Soru sormak icin POST endpoint",
            "search": "/search - Doküman aramasi icin POST endpoint", 
            "info": "/info - Sistem bilgileri icin GET endpoint",
            "health": "/health - Saglik kontrolu icin GET endpoint"
        }
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Kullanicinin sorusunu RAG sistemi uzerinden isler ve cevap doner.
    
    Args:
        request: Soru iceren request body
        
    Returns:
        QuestionResponse: Cevap ve kaynak bilgileri
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Soru bos olamaz.")
    
    try:
        rag = get_rag_system()
        result = rag.ask_question(request.question)
        
        return QuestionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Soru islenirken hata olustu: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Verilen sorguya benzer dokümanlari arar.
    
    Args:
        request: Arama sorgusu ve sonuc sayisi
        
    Returns:
        SearchResponse: Benzer dokümanlar listesi
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Arama sorgusu bos olamaz.")
    
    if request.k < 1 or request.k > 20:
        raise HTTPException(status_code=400, detail="k parametresi 1-20 arasinda olmalidir.")
    
    try:
        rag = get_rag_system()
        results = rag.search_similar_documents(request.query, request.k)
        
        return SearchResponse(results=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Doküman aramasinda hata olustu: {str(e)}"
        )


@app.get("/info", response_model=SystemInfo)
async def get_system_info():
    """
    RAG sistemi hakkinda bilgi dondurur.
    
    Returns:
        SystemInfo: Sistem bilgileri
    """
    try:
        rag = get_rag_system()
        info = rag.get_system_info()
        
        return SystemInfo(**info)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sistem bilgileri alinamadi: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """API saglik kontrolu."""
    try:
        # RAG sistemini kontrol et
        rag = get_rag_system()
        status = "healthy" if rag.qa_chain else "unhealthy"
        
        return {
            "status": status,
            "timestamp": "2024-04-13T10:51:00Z",
            "service": "CNC RAG System API",
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": "2024-04-13T10:51:00Z", 
            "service": "CNC RAG System API",
            "version": "1.0.0",
            "error": str(e)
        }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 hata yoneticisi."""
    return {
        "error": "Endpoint bulunamadi",
        "message": f"{request.url.path} adresinde bir endpoint bulunmuyor",
        "available_endpoints": ["/", "/ask", "/search", "/info", "/health"]
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """500 hata yoneticisi."""
    return {
        "error": "Sunucu hatasi",
        "message": "Sunucuda beklenmedik bir hata olustu",
        "detail": str(exc)
    }


if __name__ == "__main__":
    import uvicorn
    
    print("CNC RAG System API baslatiliyor...")
    print("API adresi: http://localhost:8000")
    print("Dokümantasyon: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
