from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class CNCRAGSystem:
    """CNC Fabrika Bakim Asistan icin RAG Sistemi"""
    
    def __init__(self, persist_directory: str = "chroma_db_v2"):
        """
        RAG sistemini initialize eder.
        
        Args:
            persist_directory: ChromaDB'nin kaydedildi dizin
        """
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.llm = OllamaLLM(model="llama3")
        self.vector_store = None
        self.qa_chain = None
        
        # Sistem promptu
        self.system_prompt = """Sen bir Fabrika Bakim Asistanisin. SADECE sana verilen Context'e dayanarak cevap ver. 
Cevap Context'te yoksa 'Bu bilgi dokumanlarda yok' de.
Context'teki bilgilere dayanarak net ve yardimci cevaplar ver."""
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Vektor deposunu ve QA zincirini yukler."""
        try:
            # ChromaDB vektor deposunu yukle
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # QA zinciri icin prompt template
            template = """{context}

Soru: {question}

{system_prompt}

Cevap:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question", "system_prompt"]
            )
            
            # LCEL zinciri olustur
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Dogru input flow icin chain yapisi
            self.qa_chain = (
                {
                    "context": lambda x: format_docs(retriever.invoke(x["question"])),
                    "question": lambda x: x["question"],
                    "system_prompt": lambda x: x["system_prompt"]
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            print("RAG sistemi basariyla yuklendi.")
            
        except Exception as e:
            print(f"RAG sistemi yuklenirken hata: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Kullanicinin sorusunu RAG sistemi uzerinden isler.
        
        Args:
            question: Kullanicinin sorusu
            
        Returns:
            Dict: Cevap ve kaynak bilgileri
        """
        if not self.qa_chain:
            return {
                "answer": "RAG sistemi henuz yuklenmedi. Once veri yukleme islemi yapin.",
                "source_documents": []
            }
        
        try:
            # Soruyu isle - LCEL zinciri otomatik olarak doküman getirir
            result = self.qa_chain.invoke({
                "question": question,
                "system_prompt": self.system_prompt
            })
            
            # Kaynak dokümanlari manuel olarak getir
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
            
            # Kaynak dokümanlarin bilgilerini formatla
            sources = []
            for doc in docs:
                source_info = {
                    "source": doc.metadata.get("source", "Bilinmeyen kaynak"),
                    "page": doc.metadata.get("page", "Bilinmeyen sayfa"),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            return {
                "answer": result,
                "source_documents": sources
            }
            
        except Exception as e:
            return {
                "answer": f"Soru islenirken hata olustu: {str(e)}",
                "source_documents": []
            }
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Verilen sorguya benzer dokumanlari arar.
        
        Args:
            query: Arama sorgusu
            k: Donecek dokuman sayisi
            
        Returns:
            List[Dict]: Benzer dokumanlar listesi
        """
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                result = {
                    "source": doc.metadata.get("source", "Bilinmeyen kaynak"),
                    "page": doc.metadata.get("page", "Bilinmeyen sayfa"),
                    "content": doc.page_content
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Dokuman aramasinda hata: {str(e)}")
            return []
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Sistem hakkinda bilgi dondurur.
        
        Returns:
            Dict: Sistem bilgileri
        """
        info = {
            "system_status": "Ready" if self.qa_chain else "Not Ready",
            "persist_directory": self.persist_directory,
            "embedding_model": "nomic-embed-text",
            "llm_model": "llama3"
        }
        
        # Vektor deposundaki dokuman sayisini kontrol et
        if self.vector_store:
            try:
                collection = self.vector_store._collection
                info["document_count"] = collection.count()
            except:
                info["document_count"] = "Unknown"
        
        return info


def create_rag_system(persist_directory: str = "chroma_db") -> CNCRAGSystem:
    """
    RAG sistemi ornegi olusturur.
    
    Args:
        persist_directory: ChromaDB dizini
        
    Returns:
        CNCRAGSystem: RAG sistemi ornegi
    """
    return CNCRAGSystem(persist_directory)


if __name__ == "__main__":
    # Test icin
    try:
        rag = create_rag_system()
        
        # Sistem bilgisini goster
        info = rag.get_system_info()
        print("=== Sistem Bilgileri ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Test sorusu
        test_question = input("\nTest sorusu girin (cikmak icin 'exit'): ")
        if test_question.lower() != 'exit':
            result = rag.ask_question(test_question)
            print(f"\nSoru: {test_question}")
            print(f"Cevap: {result['answer']}")
            
            if result['source_documents']:
                print("\nKaynaklar:")
                for i, source in enumerate(result['source_documents'], 1):
                    print(f"{i}. {source['source']} - Sayfa {source['page']}")
                    print(f"   {source['content'][:100]}...")
                    print()
    
    except Exception as e:
        print(f"Hata: {str(e)}")
        print("Once veri yukleme islemi yapmaniz gerekiyor: python -m core.ingest")
