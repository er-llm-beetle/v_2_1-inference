


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



# Import necessary libraries
import os
import logging
import json
import numpy as np
import pandas as pd
import torch
import time
import atexit
import httpx
import psutil
import openai
import pickle
import threading
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from tqdm.auto import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from functools import lru_cache

import multiprocessing as mp
from multiprocessing import current_process
import torch.multiprocessing as torch_mp
import subprocess



# Document processing
import PyPDF2
# import docx
from bs4 import BeautifulSoup
import re


# ML and embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# Vector store
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

# 
import chromadb
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
from chromadb.config import Settings



# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()







# -

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Fix common punctuation issues
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' ?', '?')
    text = text.replace(' !', '!')
    
    return text.strip()







# class LLMProcessor:
#     """Handles LLM processing for final response generation"""
    
#     def __init__(
#         self,
#         model_name: str = "unsloth/Llama-3.2-1B-Instruct",
#         device: str = None
#     ):
#         self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
#         self.model.eval()
        
#         self.logger = logging.getLogger(__name__)

#     def generate_response(
#         self,
#         query: str,
#         context: str,
#         summaries: Dict[str, str],
#         language: str = "az"
#     ) -> Dict[str, Any]:
#         """Generate final response using LLM"""
#         try:
#             # Prepare prompt
#             prompt_template = {
#                 "az": f"""Sual: {query}

#                 Məlumat mənbələri:
#                 {context}

#                 Ətraflı xülasə:
#                 {summaries.get('extractive_summary', '')}

#                 Qısa xülasə:
#                 {summaries.get('abstractive_summary', '')}

#                 Zəhmət olmasa, yuxarıdakı məlumatlar əsasında aydın və dəqiq cavab hazırlayın.
                
#                 Cavab:""",
                
#                 "en": f"""Question: {query}

#                 Information sources:
#                 {context}

#                 Detailed summary:
#                 {summaries.get('extractive_summary', '')}

#                 Brief summary:
#                 {summaries.get('abstractive_summary', '')}

#                 Please provide a clear and precise answer based on the information above.
                
#                 Answer:"""
#             }[language]

#             # Generate response
#             inputs = self.tokenizer(
#                 prompt_template,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=2048
#             ).to(self.device)

#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_length=1024,
#                     temperature=0.7,
#                     top_p=0.9,
#                     do_sample=True,
#                     pad_token_id=self.tokenizer.pad_token_id
#                 )

#             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # Clean up response by extracting only the answer part
#             answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response.strip()
            
#             return {
#                 "status": "success",
#                 "response": answer,
#                 "language": language
#             }

#         except Exception as e:
#             self.logger.error(f"Response generation failed: {str(e)}")
#             return {
#                 "status": "error",
#                 "message": str(e)
#             }

#     def __del__(self):
#         """Cleanup resources"""
#         try:
#             if hasattr(self, 'device') and self.device == "cuda":
#                 torch.cuda.empty_cache()
#         except:
#             pass



class Repacker:
    """Handles repacking of ranked results into a coherent context"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def repack_results(
        self,
        results: List[Dict],
        query: str,
        max_length: int = 4096
    ) -> Dict[str, Any]:
        """Repack ranked results into structured context"""
        try:
            if not results:
                return {
                    "context": "",
                    "sources": []
                }

            # Sort results by score
            sorted_results = sorted(results, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            
            # Build context structure
            context_parts = [
                f"Query: {query}\n\n",
                "Relevant Information:\n"
            ]
            
            sources = []
            current_length = len("".join(context_parts))
            
            for result in sorted_results:
                # Get text and clean it
                text = clean_text(result.get('text', ''))
                source_info = {
                    'text': text,
                    'score': result.get('rerank_score', 0.0),
                    'metadata': result.get('metadata', {})
                }
                
                # Check length
                if current_length + len(text) + 2 <= max_length:
                    context_parts.append(f"{text}\n\n")
                    current_length += len(text) + 2
                    sources.append(source_info)
                else:
                    break
            
            return {
                "context": "".join(context_parts).strip(),
                "sources": sources
            }
            
        except Exception as e:
            self.logger.error(f"Repacking failed: {str(e)}")
            return {
                "context": "",
                "sources": []
            }






class MilvusStorageManager:
    """Manages Milvus storage and cleanup with improved error handling"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        min_free_space_gb: float = 5.0,
        docker_compose_file: str = "docker-compose.yml"
    ):
        self.host = host
        self.port = port
        self.min_free_space_gb = min_free_space_gb
        self.docker_compose_file = docker_compose_file
        self.logger = logging.getLogger(__name__)
        
        # Initialize docker client
        try:
            import docker
            self.docker_client = docker.from_env()
        except:
            self.docker_client = None
            self.logger.warning("Docker SDK not available. Some features will be limited.")

    def check_storage_space(self) -> Tuple[float, float]:
        """Check available storage space for Milvus data"""
        try:
            if self.docker_client:
                # Get Milvus container
                containers = self.docker_client.containers.list(
                    filters={"name": "milvus-standalone"}
                )
                
                if not containers:
                    # Try to start Milvus if not running
                    self._start_milvus()
                    containers = self.docker_client.containers.list(
                        filters={"name": "milvus-standalone"}
                    )
                
                if containers:
                    # Get container stats
                    stats = containers[0].stats(stream=False)
                    
                    # Calculate space from container stats
                    total_bytes = stats['storage_stats']['total_bytes']
                    free_bytes = stats['storage_stats']['free_bytes']
                    
                    total_gb = total_bytes / (1024 * 1024 * 1024)
                    free_gb = free_bytes / (1024 * 1024 * 1024)
                    
                    return free_gb, total_gb
            
            # Fallback to host system check
            import psutil
            disk = psutil.disk_usage("/")
            return disk.free / (1024 * 1024 * 1024), disk.total / (1024 * 1024 * 1024)
            
        except Exception as e:
            self.logger.warning(f"Failed to check storage space: {str(e)}")
            # Return conservative estimates
            return 10.0, 100.0

    def _start_milvus(self) -> bool:
        """Start Milvus using docker-compose"""
        try:
            # Check if docker-compose file exists
            if not os.path.exists(self.docker_compose_file):
                self.logger.info("Downloading Milvus docker-compose file...")
                url = "https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml"
                import requests
                response = requests.get(url)
                with open(self.docker_compose_file, 'w') as f:
                    f.write(response.text)

            # Start Milvus
            subprocess.run(
                ['docker-compose', '-f', self.docker_compose_file, 'up', '-d'],
                check=True
            )
            
            # Wait for startup
            time.sleep(30)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Milvus: {str(e)}")
            return False

    def cleanup_old_collections(self):
        """Remove old collections to free up space"""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=30  # Increased timeout
            )
            
            # Get all collections
            collections = utility.list_collections()
            
            if not collections:
                return
                
            # Get collection info and sort by size/age
            collection_info = []
            for coll_name in collections:
                try:
                    coll = Collection(coll_name)
                    stats = coll.get_stats()
                    collection_info.append({
                        'name': coll_name,
                        'row_count': int(stats['row_count']),
                        'created': time.time()  # Placeholder as creation time not available
                    })
                except:
                    continue
            
            # Sort by row count (as proxy for size)
            collection_info.sort(key=lambda x: x['row_count'], reverse=True)
            
            # Remove largest collections until we free enough space
            free_space, _ = self.check_storage_space()
            for coll in collection_info:
                if free_space >= self.min_free_space_gb:
                    break
                    
                try:
                    utility.drop_collection(coll['name'])
                    self.logger.info(f"Dropped collection: {coll['name']}")
                    free_space, _ = self.check_storage_space()
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup collections: {str(e)}")
        finally:
            try:
                connections.disconnect("default")
            except:
                pass

    def ensure_storage_space(self) -> bool:
        """Ensure sufficient storage space is available"""
        free_space, total_space = self.check_storage_space()
        
        if free_space < self.min_free_space_gb:
            self.logger.warning(f"Low storage space: {free_space:.2f}GB free of {total_space:.2f}GB")
            self.cleanup_old_collections()
            
            # Check again after cleanup
            free_space, _ = self.check_storage_space()
            if free_space < self.min_free_space_gb:
                # Try recreating volumes as last resort
                if self._recreate_volumes():
                    free_space, _ = self.check_storage_space()
                
        return free_space >= self.min_free_space_gb

    def _recreate_volumes(self) -> bool:
        """Recreate Milvus volumes if needed"""
        try:
            # Stop Milvus
            subprocess.run(
                ['docker-compose', '-f', self.docker_compose_file, 'down', '-v'],
                check=True
            )
            
            # Remove old volumes
            if self.docker_client:
                volumes = self.docker_client.volumes.list(
                    filters={"name": "milvus-standalone"}
                )
                for vol in volumes:
                    vol.remove(force=True)
            
            # Start Milvus again
            return self._start_milvus()
            
        except Exception as e:
            self.logger.error(f"Failed to recreate volumes: {str(e)}")
            return False



class MilvusHealthManager:
    """Manages Milvus server health and connection stability"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        docker_compose_path: str = "./docker-compose.yml"
    ):
        self.host = host
        self.port = port
        self.docker_compose_path = docker_compose_path
        self.logger = logging.getLogger(__name__)
        
    def check_docker_installed(self) -> bool:
        """Check if Docker is installed and running"""
        try:
            result = subprocess.run(['docker', 'info'], 
                                 capture_output=True, 
                                 text=True)
            return result.returncode == 0
        except FileNotFoundError:
            self.logger.error("Docker is not installed. Please install Docker first.")
            return False

    def check_milvus_container(self) -> Tuple[bool, str]:
        """Check Milvus container status"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', 'name=milvus-standalone', '--format', '{{.Status}}'],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                return False, "not_found"
            return "Up" in result.stdout, result.stdout.strip()
        except Exception as e:
            self.logger.error(f"Failed to check Milvus container: {str(e)}")
            return False, "error"

    def start_milvus(self) -> bool:
        """Start Milvus using docker-compose"""
        try:
            # Check if docker-compose file exists
            if not os.path.exists(self.docker_compose_path):
                self.logger.info("Downloading Milvus docker-compose file...")
                url = "https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml"
                subprocess.run(['wget', url, '-O', self.docker_compose_path], check=True)

            # Start Milvus
            subprocess.run(['docker-compose', '-f', self.docker_compose_path, 'up', '-d'], check=True)
            
            # Wait for startup
            for _ in range(30):  # Wait up to 30 seconds
                running, _ = self.check_milvus_container()
                if running:
                    self.logger.info("Milvus container started successfully")
                    time.sleep(5)  # Additional wait for service readiness
                    # time.sleep(10)  # Additional wait for service readiness
                    return True
                time.sleep(1)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start Milvus: {str(e)}")
            return False

    def restart_milvus(self) -> bool:
        """Restart Milvus container"""
        try:
            subprocess.run(['docker-compose', '-f', self.docker_compose_path, 'restart'], check=True)
            time.sleep(15)  # Wait for restart
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart Milvus: {str(e)}")
            return False

    def ensure_milvus_running(self) -> bool:
        """Ensure Milvus is running and healthy"""
        if not self.check_docker_installed():
            return False

        running, status = self.check_milvus_container()
        
        if not running:
            if "Exited" in status:
                self.logger.info("Milvus container found but not running. Attempting restart...")
                return self.restart_milvus()
            else:
                self.logger.info("Milvus container not found. Attempting to start...")
                return self.start_milvus()
                
        return True




class MilvusConnectionManager:
    """Manages Milvus connections with health monitoring"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        self.health_manager = MilvusHealthManager(host, port)
        
    def connect(self) -> bool:
        """Establish connection with health check"""
        if not self.health_manager.ensure_milvus_running():
            raise ConnectionError("Failed to ensure Milvus is running")

        for attempt in range(self.max_retries):
            try:
                # Disconnect if already connected
                try:
                    connections.disconnect("default")
                except:
                    pass

                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    timeout=30
                )
                
                # Test connection
                utility.list_collections()
                
                self.logger.info(f"Successfully connected to Milvus at {self.host}:{self.port}") ## check it
                return True

            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    # Try to fix Milvus if needed
                    self.health_manager.ensure_milvus_running()
                    time.sleep(self.retry_delay)

        self.logger.error("Failed to connect to Milvus after all retries")
        return False

    def ensure_connection(self) -> bool:
        """Ensure connection is alive and reconnect if needed"""
        try:
            # Test connection
            utility.list_collections()
            return True
        except:
            return self.connect()

    def disconnect(self):
        """Safely disconnect"""
        try:
            connections.disconnect("default")
        except:
            pass







@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    source: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    word_count: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    semantic_density: Optional[float] = None




class DocumentProcessor:
    """Handles document reading and initial processing"""
    
    # Define supported formats as a class attribute
    SUPPORTED_FORMATS = {'.pdf', '.txt', '.html', '.docx', '.doc'}

    def __init__(self):
        """Initialize document processor"""
        # Create instance copy of supported formats
        self.supported_formats = self.SUPPORTED_FORMATS.copy()
                
    def read_document(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Read document and return content with metadata"""
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file format
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: {', '.join(self.supported_formats)}")
            
        # Create basic metadata
        metadata = {
            "source": str(file_path),
            "file_type": file_path.suffix,
            "file_name": file_path.name,
            "processed_at": datetime.now().isoformat(),
            "file_size": file_path.stat().st_size
        }
        
        # Read content based on file type
        try:
            if file_path.suffix == '.pdf':
                content = self._read_pdf(file_path)
                text = content["text"]
                metadata.update({k: v for k, v in content.items() if k != "text"})
            elif file_path.suffix == '.html':
                text = self._read_html(file_path)
            elif file_path.suffix in {'.docx', '.doc'}:
                raise NotImplementedError(f"Handler for {file_path.suffix} is not yet implemented")
            else:  # .txt
                text = self._read_text(file_path)
                
            # Clean text
            text = self._clean_text(text)
            
            # Add content statistics to metadata
            metadata.update({
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.splitlines())
            })
            
            return text, metadata
            
        except Exception as e:
            raise RuntimeError(f"Error reading document: {str(e)}")

    def _read_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Read content from PDF file"""
        text = []
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["page_count"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    content = page.extract_text()
                    if content:
                        text.append(content)
                        
            return {
                "text": "\n\n".join(text),
                "page_count": metadata["page_count"]
            }
            
        except Exception as e:
            raise RuntimeError(f"Error reading PDF: {str(e)}")

    def _read_text(self, file_path: Path) -> str:
        """Read content from text file"""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try different encodings if UTF-8 fails
            for encoding in ['utf-8-sig', 'cp1251', 'latin1', 'ascii']:
                try:
                    return file_path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise RuntimeError("Could not decode text file with any supported encoding")
            
    def _read_html(self, file_path: Path) -> str:
        """Read content from HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                # Get text and clean it
                text = soup.get_text(separator='\n')
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return '\n'.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            raise RuntimeError(f"Error reading HTML: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
            
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])\s+', r'\1 ', text)
        
        # Handle special characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()

    def __str__(self):
        """String representation"""
        return f"DocumentProcessor(supported_formats={self.supported_formats})"


# for milvus:
# class ChunkingStrategy:
#     """Document-aware chunking strategy optimized for policy documents"""
    
#     def __init__(
#         self,
#         chunk_size: int = 1024,        # Larger for policy documents
#         chunk_overlap: int = 128,       # Increased overlap for context
#         min_chunk_size: int = 512,      # Increased minimum
#         max_chunk_size: int = 60000     # Milvus limit
#     ):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.min_chunk_size = min_chunk_size
#         self.max_chunk_size = max_chunk_size
        
#         # Document structure markers
#         self.section_markers = [
#             "# ",           # Main headers
#             "## ",          # Sub headers
#             "### ",         # Sub-sub headers
#             "PROSEDUR",
#             "QAYDALAR",
#             "Ümumi müddəalar",
#             "Yekun müddəalar",
#             "ANLAYIŞLAR",
#             "ƏHATƏ DAİRƏSİ",
#             "SİYASƏT"
#         ]
        
#         # Initialize splitters
#         self.semantic_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=[
#                 "\n\n\n",     # Multiple line breaks (strong section boundary)
#                 "\n\n",       # Paragraph boundary
#                 "\n",         # Line boundary
#                 ".",          # Sentence boundary
#                 ";",          # Clause boundary
#                 ",",          # Phrase boundary
#                 " ",          # Word boundary
#                 ""           # Character boundary
#             ],
#             length_function=len,
#         )

#     def _is_section_boundary(self, text: str) -> bool:
#         """Check if text starts with a section marker"""
#         text = text.strip()
#         return any(text.startswith(marker) for marker in self.section_markers)

#     def _clean_chunk_text(self, text: str) -> str:
#         """Clean and normalize chunk text"""
#         # Replace tabs with spaces
#         text = text.replace('\t', ' ')
        
#         # Fix spacing around punctuation
#         text = re.sub(r'\s+([.,;!?])', r'\1', text)
        
#         # Normalize whitespace
#         text = ' '.join(text.split())
        
#         # Ensure proper sentence spacing
#         text = re.sub(r'([.!?])\s*([A-ZƏÜÖŞÇĞIİ])', r'\1 \2', text)
        
#         return text.strip()

#     def _split_sections(self, text: str) -> List[str]:
#         """Split text into major sections"""
#         sections = []
#         current_section = []
        
#         for line in text.split('\n'):
#             if self._is_section_boundary(line):
#                 if current_section:
#                     sections.append('\n'.join(current_section))
#                 current_section = [line]
#             else:
#                 current_section.append(line)
        
#         if current_section:
#             sections.append('\n'.join(current_section))
            
#         return sections

#     def create_chunks(
#         self,
#         text: str,
#         metadata: Dict[str, Any]
#     ) -> List[Dict[str, Any]]:
#         """Create document-aware chunks"""
#         chunks = []
#         chunk_index = 0
        
#         # Split into major sections first
#         sections = self._split_sections(text)
        
#         for section in sections:
#             # Skip empty sections
#             if not section.strip():
#                 continue
            
#             # Clean section text
#             section = self._clean_chunk_text(section)
            
#             # Detect section title
#             section_title = None
#             for marker in self.section_markers:
#                 if section.startswith(marker):
#                     section_title = section.split('\n')[0].strip()
#                     break
            
#             # Create chunks for this section
#             if len(section) <= self.max_chunk_size:
#                 # Small enough to be a single chunk
#                 chunk_metadata = ChunkMetadata(
#                     source=metadata["source"],
#                     chunk_index=chunk_index,
#                     total_chunks=len(sections),
#                     start_char=text.find(section),
#                     end_char=text.find(section) + len(section),
#                     word_count=len(section.split()),
#                     section_title=section_title,
#                     semantic_density=self._calculate_semantic_density(section)
#                 )
#                 chunks.append({
#                     "text": section,
#                     "metadata": chunk_metadata
#                 })
#                 chunk_index += 1
#             else:
#                 # Split large sections into smaller chunks
#                 section_chunks = self.semantic_splitter.create_documents([section])
#                 for sub_chunk in section_chunks:
#                     chunk_text = self._clean_chunk_text(sub_chunk.page_content)
#                     if len(chunk_text) >= self.min_chunk_size:
#                         chunk_metadata = ChunkMetadata(
#                             source=metadata["source"],
#                             chunk_index=chunk_index,
#                             total_chunks=len(sections),
#                             start_char=text.find(chunk_text),
#                             end_char=text.find(chunk_text) + len(chunk_text),
#                             word_count=len(chunk_text.split()),
#                             section_title=section_title,
#                             semantic_density=self._calculate_semantic_density(chunk_text)
#                         )
#                         chunks.append({
#                             "text": chunk_text,
#                             "metadata": chunk_metadata
#                         })
#                         chunk_index += 1
        
#         return self._deduplicate_chunks(chunks)

#     def _calculate_semantic_density(self, text: str) -> float:
#         """Calculate semantic density of text"""
#         words = text.split()
#         unique_words = set(words)
        
#         if not words:
#             return 0.0
            
#         # Calculate lexical diversity
#         lexical_diversity = len(unique_words) / len(words)
        
#         # Calculate punctuation density
#         punctuation_count = sum(1 for char in text if char in '.,;:!?()[]{}')
#         punctuation_density = punctuation_count / len(words)
        
#         # Combine metrics
#         return (0.7 * lexical_diversity + 0.3 * punctuation_density)

#     def _deduplicate_chunks(
#         self,
#         chunks: List[Dict[str, Any]],
#         similarity_threshold: float = 0.9
#     ) -> List[Dict[str, Any]]:
#         """Remove duplicate or highly similar chunks"""
#         unique_chunks = []
#         seen_texts = set()
        
#         for chunk in chunks:
#             text = chunk["text"].strip()
            
#             # Quick exact duplicate check
#             if text not in seen_texts:
#                 seen_texts.add(text)
#                 unique_chunks.append(chunk)
        
#         return unique_chunks


# for chromadb:
class ChunkingStrategy:
    """Document-aware chunking strategy optimized for policy documents"""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        min_chunk_size: int = 512,
        max_chunk_size: int = 60000
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Document structure markers
        self.section_markers = [
            "# ",
            "## ",
            "### ",
            "PROSEDUR",
            "QAYDALAR",
            "Ümumi müddəalar",
            "Yekun müddəalar",
            "ANLAYIŞLAR",
            "ƏHATƏ DAİRƏSİ",
            "SİYASƏT"
        ]
        
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ".",
                ";",
                ",",
                " ",
                ""
            ],
            length_function=len,
        )

    def _is_section_boundary(self, text: str) -> bool:
        """Check if text starts with a section marker"""
        text = text.strip()
        return any(text.startswith(marker) for marker in self.section_markers)

    def _create_chunk_metadata(
        self,
        text: str,
        source: str,
        chunk_index: int,
        total_chunks: int,
        full_text: str,
        section_title: str = None
    ) -> dict:
        """Create metadata dictionary for a chunk"""
        words = text.split()
        unique_words = set(words)
        semantic_density = len(unique_words) / len(words) if words else 0
        
        return {
            'source': source,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'start_char': full_text.find(text),
            'end_char': full_text.find(text) + len(text),
            'word_count': len(words),
            'section_title': section_title,
            'semantic_density': semantic_density
        }

    def create_chunks(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create document-aware chunks"""
        try:
            chunks = []
            chunk_index = 0
            
            # Split into major sections first
            sections = self._split_sections(text)
            
            for section in sections:
                if not section.strip():
                    continue
                
                # Clean section text
                section = self._clean_chunk_text(section)
                
                # Detect section title
                section_title = None
                for marker in self.section_markers:
                    if section.startswith(marker):
                        section_title = section.split('\n')[0].strip()
                        break
                
                # Handle section based on size
                if len(section) <= self.max_chunk_size:
                    chunk_metadata = self._create_chunk_metadata(
                        text=section,
                        source=metadata["source"],
                        chunk_index=chunk_index,
                        total_chunks=len(sections),
                        full_text=text,
                        section_title=section_title
                    )
                    
                    chunks.append({
                        "text": section,
                        "metadata": chunk_metadata
                    })
                    chunk_index += 1
                else:
                    # Split large sections
                    section_chunks = self.semantic_splitter.create_documents([section])
                    for sub_chunk in section_chunks:
                        chunk_text = self._clean_chunk_text(sub_chunk.page_content)
                        if len(chunk_text) >= self.min_chunk_size:
                            chunk_metadata = self._create_chunk_metadata(
                                text=chunk_text,
                                source=metadata["source"],
                                chunk_index=chunk_index,
                                total_chunks=len(sections),
                                full_text=text,
                                section_title=section_title
                            )
                            
                            chunks.append({
                                "text": chunk_text,
                                "metadata": chunk_metadata
                            })
                            chunk_index += 1
            
            return self._deduplicate_chunks(chunks)
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise

    def _split_sections(self, text: str) -> List[str]:
        """Split text into major sections"""
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            if self._is_section_boundary(line):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
            
        return sections

    def _clean_chunk_text(self, text: str) -> str:
        """Clean and normalize chunk text"""
        text = text.replace('\t', ' ')
        text = ' '.join(text.split())
        text = re.sub(r'\s+([.,;!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-ZƏÜÖŞÇĞIİ])', r'\1 \2', text)
        return text.strip()

    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density of text"""
        words = text.split()
        unique_words = set(words)
        
        if not words:
            return 0.0
            
        lexical_diversity = len(unique_words) / len(words)
        punctuation_count = sum(1 for char in text if char in '.,;:!?()[]{}')
        punctuation_density = punctuation_count / len(words)
        
        return (0.7 * lexical_diversity + 0.3 * punctuation_density)

    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        similarity_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """Remove duplicate or highly similar chunks"""
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            text = chunk["text"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_chunks.append(chunk)
        
        return unique_chunks










# Add ModelCache class (new)
class ModelCache:
    """Singleton cache for the embedding model"""
    _instance = None
    _lock = threading.Lock()
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def get_model(
        self,
        model_name: str,
        device: str = None,
        cache_dir: str = "model_cache"
    ) -> SentenceTransformer:
        """Get model from cache or load it"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model(model_name, device, cache_dir)
        return self._model
    
    def _load_model(
        self,
        model_name: str,
        device: str = None,
        cache_dir: str = "model_cache"
    ) -> SentenceTransformer:
        """Load model with caching"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = Path(cache_dir) / f"{model_name.replace('/', '_')}.pt"
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            if cache_path.exists():
                logger.info(f"Loading model from cache: {cache_path}")
                model = SentenceTransformer(model_name, device=device)
                model.load_state_dict(torch.load(cache_path, map_location=device))
            else:
                logger.info(f"Loading model from scratch: {model_name}")
                model = SentenceTransformer(model_name, device=device)
                torch.save(model.state_dict(), cache_path)
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

# Add ChunkCache class (new)
class ChunkCache:
    """Cache for document chunks"""
    def __init__(self, cache_dir: str = "chunk_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "chunks.pkl"
        self.metadata_file = self.cache_dir / "chunks_metadata.json"
        self._lock = threading.Lock()
        
    def _get_document_hash(self, file_path: str) -> str:
        """Generate hash of document content"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_chunks(self, doc_path: str) -> Optional[List[Dict[str, Any]]]:
        """Get chunks from cache if available"""
        try:
            if not self.cache_file.exists() or not self.metadata_file.exists():
                return None
            
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            current_hash = self._get_document_hash(doc_path)
            if metadata.get('doc_hash') != current_hash:
                return None
            
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.warning(f"Error reading chunk cache: {str(e)}")
            return None
    
    def save_chunks(self, chunks: List[Dict[str, Any]], doc_path: str):
        """Save chunks to cache"""
        try:
            with self._lock:
                # Save chunks
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(chunks, f)
                
                # Save metadata
                metadata = {
                    'doc_hash': self._get_document_hash(doc_path),
                    'created_at': datetime.now().isoformat(),
                    'chunk_count': len(chunks)
                }
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f)
                    
        except Exception as e:
            logger.error(f"Error saving chunk cache: {str(e)}")

class EmbeddingCache:
    """Cache for document embeddings"""
    def __init__(
        self,
        cache_dir: str = "embedding_cache",
        max_memory_items: int = 10000,
        ttl_days: int = 30
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "embedding_index.json"
        self.ttl = timedelta(days=ttl_days)
        self._lock = threading.Lock()
        
        # Initialize memory cache
        self._memory_cache = lru_cache(maxsize=max_memory_items)(self._get_embedding)
        self._load_index()
    
    def _load_index(self):
        """Load cache index"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}
        except Exception as e:
            logger.warning(f"Error loading cache index: {str(e)}")
            self.index = {}
    
    def _save_index(self):
        """Save cache index"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.warning(f"Error saving cache index: {str(e)}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from disk cache"""
        cache_key = self._get_cache_key(text)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        
        try:
            if cache_path.exists():
                return np.load(cache_path)
            return None
        except Exception:
            return None
    
    def get_embeddings(
        self,
        texts: List[str]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Get embeddings from cache"""
        cached_embeddings = []
        texts_to_compute = []
        
        for text in texts:
            # Try memory cache first
            embedding = self._memory_cache(text)
            if embedding is not None:
                cached_embeddings.append(embedding)
            else:
                texts_to_compute.append(text)
        
        return cached_embeddings, texts_to_compute
    
    def save_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray
    ):
        """Save embeddings to cache"""
        try:
            for text, embedding in zip(texts, embeddings):
                cache_key = self._get_cache_key(text)
                cache_path = self.cache_dir / f"{cache_key}.npy"
                
                # Save to disk
                np.save(cache_path, embedding)
                
                # Update index
                self.index[cache_key] = {
                    'created_at': datetime.now().isoformat(),
                    'text_hash': hashlib.md5(text.encode('utf-8')).hexdigest()
                }
                
                # Update memory cache
                self._memory_cache(text)
            
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    def cleanup(self):
        """Remove expired cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for cache_key, info in self.index.items():
                created_at = datetime.fromisoformat(info['created_at'])
                if current_time - created_at > self.ttl:
                    expired_keys.append(cache_key)
                    cache_path = self.cache_dir / f"{cache_key}.npy"
                    if cache_path.exists():
                        cache_path.unlink()
            
            for key in expired_keys:
                del self.index[key]
            
            self._save_index()
            
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {str(e)}")






# class EmbeddingModel:
#     """Memory-optimized embedding model"""
    
#     def __init__(
#         self,
#         # model_name: str = "BAAI/bge-m3",
#         model_name: str = "LocalDoc/TEmA-small",
#         device: str = None,
#         max_length: int = 512
#     ):
#         self.max_length = max_length
#         if device is None:
#             if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 4 * 1024 * 1024 * 1024:  # 4GB
#                 self.device = "cuda"
#             else:
#                 self.device = "cpu"
#             # self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = device
            
#         # Load model with memory optimization
#         try:
#             # Set memory efficient attention
#             torch.backends.cuda.max_split_size_mb = 512  # Limit CUDA memory split size
            
#             self.model = SentenceTransformer(
#                 model_name,
#                 device=self.device
#             )
#             # Enable memory optimization
#             self.model.half()  # Convert to FP16 to save memory
#             torch.cuda.empty_cache()  # Clear CUDA cache
#         except Exception as e:
#             logger.error(f"Failed to load model: {str(e)}")
#             raise
        
#         self.dimension = self.model.get_sentence_embedding_dimension()
    
#     def generate_embeddings(
#         self,
#         texts: List[str],
#         batch_size: int = 4  # Smaller batch size to manage memory
#     ) -> np.ndarray:
#         """Generate embeddings with memory management"""
#         embeddings = []
        
#         try:
#             # Process in smaller batches
#             for i in tqdm(range(0, len(texts), batch_size)):
#                 batch = texts[i:i + batch_size]

#                 # Clear cache before processing each batch
#                 if self.device == "cuda":
#                     torch.cuda.empty_cache()

#                 with torch.no_grad():  # Disable gradient computation
#                     batch_embeddings = self.model.encode(
#                         batch,
#                         normalize_embeddings=True,
#                         show_progress_bar=False,
#                         convert_to_numpy=True
#                     )
#                 embeddings.extend(batch_embeddings)
                
#                 # Clear memory after each batch
#                 if self.device == "cuda":
#                     torch.cuda.empty_cache()
                    
#             return np.array(embeddings)
            
#         except Exception as e:
#             logger.error(f"Embedding generation failed: {str(e)}")
#             raise

import requests


class HuggingFaceEmbeddingModel:
    """Embedding model using Hugging Face's Inference API with simple requests"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        api_key: str = None,
        device: str = None,
        batch_size: int = 8,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dimension = 1024
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), disable=not show_progress):
            batch = texts[i:i + self.batch_size]
            
            # Process each text individually to get its embedding
            for text in batch:
                # Create payload for single text
                payload = {
                    "inputs": text,
                    "options": {"wait_for_model": True}
                }
                
                # Get embedding for single text
                embedding = self._get_embedding(payload)
                if embedding is not None:
                    all_embeddings.append(embedding)
        
        if not all_embeddings:
            raise RuntimeError("Failed to generate any embeddings")
            
        return np.array(all_embeddings)
    
    def _get_embedding(self, payload: dict) -> Optional[np.ndarray]:
        """Get embedding for single text with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    embedding_data = response.json()
                    # Get the first embedding (should only be one)
                    if isinstance(embedding_data, list) and embedding_data:
                        return np.array(embedding_data[0])
                else:
                    self.logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                    if "Model is loading" in response.text:
                        # Wait longer if model is loading
                        time.sleep(20)
                    elif attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        self.logger.error(f"Failed to get embedding after {self.max_retries} attempts")
        return None
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return self.dimension
    
    def __del__(self):
        """Cleanup resources"""
        pass



# Add new CachedEmbeddingModel (replaces old version)
class CachedEmbeddingModel:
    """Embedding model with comprehensive caching"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = None,
        cache_dir: str = "cache",
        max_memory_items: int = 10000,
        ttl_days: int = 30
    ):
        # Initialize caches
        self.model_cache = ModelCache()
        self.embedding_cache = EmbeddingCache(
            cache_dir=f"{cache_dir}/embeddings",
            max_memory_items=max_memory_items,
            ttl_days=ttl_days
        )
        
        # Load model through cache
        self.model = self.model_cache.get_model(
            model_name,
            device=device,
            cache_dir=f"{cache_dir}/model"
        )
        
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings with caching"""
        if not texts:
            return np.array([])
        
        # Get cached embeddings
        cached_embeddings, texts_to_compute = self.embedding_cache.get_embeddings(texts)
        
        # Generate missing embeddings
        if texts_to_compute:
            try:
                # Process in batches
                new_embeddings = []
                for i in range(0, len(texts_to_compute), batch_size):
                    batch = texts_to_compute[i:i + batch_size]
                    
                    with torch.no_grad():
                        batch_embeddings = self.model.encode(
                            batch,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            batch_size=batch_size
                        )
                    new_embeddings.extend(batch_embeddings)
                
                # Save new embeddings to cache
                self.embedding_cache.save_embeddings(texts_to_compute, new_embeddings)
                
                # Combine with cached embeddings
                all_embeddings = cached_embeddings + new_embeddings
                
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise
        else:
            all_embeddings = cached_embeddings
        
        return np.array(all_embeddings)
    
    def cleanup(self):
        """Clean up caches"""
        self.embedding_cache.cleanup()






class GPTEmbeddingModel:
    """GPT-powered embedding model"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = None,
        device: str = None,  # Kept for compatibility but not used
        batch_size: int = 100,  # OpenAI recommends max 100 texts per batch
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize OpenAI client
        if api_key:
            # self.client = OpenAI(api_key=api_key, http_client=httpx.Client(http2=True, verify=False))
            self.client = OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = OpenAI()
            
        # Set embedding dimensions based on model
        self.dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            # "text-embedding-ada-002": 1536  # Legacy model
        }

        self.dimension = self.dimension_map.get(model_name, 1536)
        
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for given texts using OpenAI's API"""
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), disable=not show_progress):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._process_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def _process_batch(self, batch: List[str]) -> List[np.ndarray]:
        """Process a batch of texts with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                # Extract embeddings from response
                embeddings = [np.array(data.embedding) for data in response.data]
                return embeddings
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
                    raise
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension for compatibility with existing code"""
        return self.dimension

    def __del__(self):
        """Cleanup resources"""
        pass  # No cleanup needed for API-based model









# class OptimizedMilvusVectorStore:
class MilvusVectorStore:
    """Handles vector storage and retrieval using Milvus"""
    
    def __init__(
        self,
        collection_name: str = "document_store",
        host: str = "localhost",
        port: int = 19530,
        embedding_dim: int = 1024,
        max_text_length: int = 32766  # Safe limit for Milvus
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.max_text_length = max_text_length

        # Default optimized index parameters
        self.index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_SQ8",  # Changed from IVF_FLAT for better performance
            "params": {
                "nlist": 1024,  # Increased from 128 for better partitioning
                "nprobe": 16    # Added search probe parameter
            }
        }



# 
        # Initialize managers
        self.storage_manager = MilvusStorageManager(host, port)
        self.connection_manager = MilvusConnectionManager(host, port)

        # Check storage space
        if not self.storage_manager.ensure_storage_space():
            raise RuntimeError("Insufficient storage space for Milvus")
            
        # Establish connection
        if not self.connection_manager.connect():
            raise ConnectionError("Failed to establish connection with Milvus")
            
        # Initialize connection pool
        self._init_connection_pool()
        

# cache
    def _create_metadata_dict(self, metadata) -> dict:
        """Convert ChunkMetadata to dictionary"""
        if isinstance(metadata, dict):
            return metadata
        return {
            'source': metadata.source,
            'chunk_index': metadata.chunk_index,
            'total_chunks': metadata.total_chunks,
            'start_char': metadata.start_char,
            'end_char': metadata.end_char,
            'word_count': metadata.word_count,
            'page_number': metadata.page_number,
            'section_title': metadata.section_title,
            'semantic_density': metadata.semantic_density
        }
# 



    def _init_connection_pool(self):
        """Initialize connection pool for better performance"""
        try:
            import multiprocessing
            self.num_connections = min(multiprocessing.cpu_count(), 4)  # Limit pool size
            
            for _ in range(self.num_connections):
                if not self.connection_manager.connect():
                    raise ConnectionError("Failed to establish connection with Milvus")
                    
            # Create and load collection
            self._create_collection()
            self.collection.load()  # Preload collection into memory
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise



# 


# # Remove initial version --
#         # Ensure connection
#         try:
#             # Disconnect if already connected
#             try:
#                 connections.disconnect("default")
#             except:
#                 pass
                
#             # Connect to Milvus
#             connections.connect(
#                 alias="default",
#                 host=host,
#                 port=port,
#                 timeout=30
#             )
#             logger.info(f"Successfully connected to Milvus at {host}:{port}")
#         except Exception as e:
#             logger.error(f"Failed to connect to Milvus: {str(e)}")
#             raise
            
#         self._create_collection()
# # Remove initial version --


    def _create_collection(self):
        """Create Milvus collection if it doesn't exist"""
        try:
            # Check existing collections
            existing_collections = utility.list_collections()
            
            # Drop existing collection if needed
            if self.collection_name in existing_collections:
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}") ## check it to remove
            
            # Create fields
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=self.max_text_length),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document store for RAG pipeline"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            logger.info(f"Created new collection: {self.collection_name}")
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.collection.create_index(
                field_name="embeddings",
                index_params=index_params
            )
            logger.info("Created index on embeddings field")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit Milvus varchar limit"""
        if len(text) > self.max_text_length:
            return text[:self.max_text_length-3] + "..."
        return text

    def insert(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict],
        batch_size: int = 100
    ):
        """Insert documents with batching and error handling"""
        try:
            # Input validation
            if not texts or not embeddings.size or len(texts) != len(embeddings) or len(texts) != len(metadata):
                raise ValueError("Invalid input: texts, embeddings, and metadata must have the same length")

            # Check storage space before processing
            if not self.storage_manager.ensure_storage_space():
                logger.warning("Low storage space. Attempting cleanup...")
                if not self.storage_manager.ensure_storage_space():
                    raise RuntimeError("Insufficient storage space after cleanup")
          

            # Prepare data
            processed_texts = []
            processed_metadata = []
            processed_embeddings = []
            
            logger.info(f"Processing {len(texts)} documents for insertion")
            
            # Process each document
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                # Truncate text if needed
                processed_text = self._truncate_text(text)
                
                # Update metadata if text was truncated
                # meta_copy = meta.copy() if meta else {}
                # if len(text) > self.max_text_length:
                #     meta_copy['full_text'] = text
                #     meta_copy['is_truncated'] = True
                #     meta_copy['original_length'] = len(text)

                # Convert metadata to dictionary and update if needed
                meta_dict = self._create_metadata_dict(meta)
                if len(text) > self.max_text_length:
                    meta_dict['full_text'] = text
                    meta_dict['is_truncated'] = True
                    meta_dict['original_length'] = len(text)


                processed_texts.append(processed_text)
                # processed_metadata.append(meta_copy)
                processed_metadata.append(meta_dict) # cache version
            
                processed_embeddings.append(embedding.tolist())
            
            # Insert in batches
            total_batches = (len(processed_texts) + batch_size - 1) // batch_size

            for i in range(0, len(processed_texts), batch_size):
                end_idx = min(i + batch_size, len(processed_texts))
                batch_num = i // batch_size + 1
                
                logger.info(f"Inserting batch {batch_num}/{total_batches}")
                
                batch_data = [
                    processed_texts[i:end_idx],
                    processed_embeddings[i:end_idx],
                    processed_metadata[i:end_idx]
                ]
                
                # try:
                #     self.collection.insert(batch_data)
                #     self.collection.flush()
                # except Exception as e:
                #     logger.error(f"Failed to insert batch {batch_num}/{total_batches}: {str(e)}")
                #     continue

#                
                import time
               
                success = False
                for attempt in range(3):
                    try:
                        # Check storage and connection before each batch
                        if not self.storage_manager.ensure_storage_space():
                            raise RuntimeError("Insufficient storage space")
                            
                        if not self.connection_manager.ensure_connection():
                            raise ConnectionError("Lost connection to Milvus")
                            
                        self.collection.insert(batch_data)
                        # self.collection.flush()
                        self.collection.flush(async_flush=True)
                        
                        logger.info(f"Successfully inserted batch {batch_num}/{total_batches}")
                        success = True
                        break
                        
                    except Exception as e:
                        logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed: {str(e)}")
                        time.sleep(2)
                
                if not success:
                    raise RuntimeError(f"Failed to insert batch {batch_num} after all retries")
            
            logger.info("Successfully completed all insertions")
      
# 
            # Ensure index exists
            try:
                print('connect.......')
                if not self.collection.has_index():
                    logger.info("Creating index as it doesn't exist")
                    self.collection.create_index(
                        field_name="embeddings",
                        index_params={
                            "metric_type": "COSINE",
                            "index_type": "IVF_FLAT",
                            "params": {"nlist": 128}
                        }
                    )
            except Exception as e:
                logger.warning(f"Index check/creation warning: {str(e)}")
            
            logger.info("Successfully completed insertion")
            
        except Exception as e:
            logger.error(f"Insert operation failed: {str(e)}")
            raise

    # def clean_text(text: str) -> str:
    #     """Clean text by removing extra whitespace and special characters"""
    #     # Replace tabs with spaces
    #     text = text.replace('\t', ' ')
        
    #     # Replace multiple spaces with single space
    #     text = ' '.join(text.split())
        
    #     # Fix common punctuation issues
    #     text = text.replace(' .', '.')
    #     text = text.replace(' ,', ',')
    #     text = text.replace(' ?', '?')
    #     text = text.replace(' !', '!')
        
    #     return text.strip()


    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        limit: int = 5
    ) -> List[Dict]:
        """Perform hybrid search with error handling"""
        try:
            logger.info("Starting hybrid search")
            
            # Load collection
            try:
                self.collection.load()
                logger.info("Collection loaded successfully")
            except Exception as e:
                logger.warning(f"Collection load warning (continuing anyway): {str(e)}")
            
            # Optimized search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {
                    "nprobe": min(16, self.index_params["params"]["nlist"] // 8),
                    "ef": 64  # Added for HNSW index type if used
                }
            }
            
            
            # Perform search
            results = []
            search_results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embeddings",
                param=search_params,
                limit=limit,
                output_fields=["text", "metadata"],
                round_decimal=-1  # Disable score rounding for speed
            )
            
            logger.info(f"Found {len(search_results[0])} search results")
            
            # Process results
            for hits in search_results:
                for hit in hits:
                    try:
                        # Get complete document
                        docs = self.collection.query(
                            expr=f"id == {hit.id}",
                            output_fields=["text", "metadata"]
                        )
                        
                        if not docs:
                            continue
                            
                        doc = docs[0]
                        # text = doc.get("text", "")
                        text = clean_text(doc.get("text", ""))
                        metadata = doc.get("metadata", {})
                        
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                        
                        # Get full text if it was truncated
                        if metadata.get("is_truncated", False):
                            text = metadata.get("full_text", text)
                        
                        results.append({
                            "id": hit.id,
                            "text": text,
                            "score": float(hit.score),
                            "metadata": metadata
                        })
                    except Exception as e:
                        logger.warning(f"Failed to process search result: {str(e)}")
                        continue
            
            logger.info(f"Successfully processed {len(results)} results")
            print("\nSearch results: ", results)
            return results
            
        except Exception as e:
            logger.error(f"Search operation failed: {str(e)}")
            raise
        finally:
            # Always try to release collection
            try:
                self.collection.release()
            except:
                pass

    def __del__(self):
        """Cleanup resources"""
        try:
            # Release collection
            if hasattr(self, 'collection'):
                try:
                    self.collection.release()
                except:
                    pass
            
            # Disconnect from Milvus
            try:
                connections.disconnect("default")
                # self.connection_manager.disconnect() # uncomment it
            except:
                pass
        except:
            pass










# class ChromaDBVectorStore:
#     """Handles vector storage and retrieval using ChromaDB"""
    
#     def __init__(
#         self,
#         collection_name: str = "document_store",
#         embedding_dim: int = 1024,
#         persist_directory: str = "chromadb_data"
#     ):
#         try:
#             self.collection_name = collection_name
#             self.embedding_dim = embedding_dim
#             self.logger = logging.getLogger(__name__)
            
#             # Initialize ChromaDB client with new configuration
#             self.client = chromadb.PersistentClient(
#                 path=persist_directory,
#                 settings=Settings(
#                     allow_reset=True,
#                     anonymized_telemetry=False
#                 )
#             )
            
#             # Get or create collection
#             self.collection = self._create_collection()
#         except Exception as e:
#             self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
#             raise

#     def _create_collection(self):
#         """Create ChromaDB collection if it doesn't exist"""
#         try:
#             # Delete collection if it exists
#             try:
#                 self.client.delete_collection(self.collection_name)
#             except Exception as e:
#                 self.logger.warning(f"No existing collection to delete: {str(e)}")
            
#             # Create new collection
#             collection = self.client.create_collection(
#                 name=self.collection_name,
#                 metadata={"hnsw:space": "cosine"},
#                 embedding_function=None
#             )
            
#             return collection
            
#         except Exception as e:
#             self.logger.error(f"Failed to create collection: {str(e)}")
#             raise

#     def _sanitize_metadata_value(self, value: Any) -> Any:
#         """Convert metadata values to ChromaDB-compatible types"""
#         if value is None:
#             return ""
#         elif isinstance(value, (str, int, float, bool)):
#             return value
#         elif isinstance(value, (list, tuple)):
#             return str(list(value))
#         elif isinstance(value, dict):
#             return str(value)
#         elif hasattr(value, '__dict__'):
#             # Handle objects with __dict__ attribute
#             return str(value.__dict__)
#         else:
#             return str(value)

#     def _convert_metadata_to_dict(self, metadata: Any) -> Dict:
#         """Convert metadata to ChromaDB-compatible dictionary format"""
#         try:
#             # If metadata is already a dict, use it as base
#             if isinstance(metadata, dict):
#                 base_dict = metadata
#             # If metadata has __dict__, use that
#             elif hasattr(metadata, '__dict__'):
#                 base_dict = metadata.__dict__
#             # If metadata is a ChunkMetadata object, extract its attributes
#             elif isinstance(metadata, ChunkMetadata):
#                 base_dict = {
#                     'source': metadata.source,
#                     'chunk_index': metadata.chunk_index,
#                     'total_chunks': metadata.total_chunks,
#                     'start_char': metadata.start_char,
#                     'end_char': metadata.end_char,
#                     'word_count': metadata.word_count,
#                     'page_number': metadata.page_number,
#                     'section_title': metadata.section_title,
#                     'semantic_density': metadata.semantic_density
#                 }
#             else:
#                 base_dict = {"value": str(metadata)}

#             # Sanitize all values in the dictionary
#             return {
#                 k: self._sanitize_metadata_value(v)
#                 for k, v in base_dict.items()
#             }
            
#         except Exception as e:
#             self.logger.warning(f"Error converting metadata to dict: {str(e)}")
#             return {"error": "Failed to convert metadata"}

#     def insert(
#         self,
#         texts: List[str],
#         embeddings: np.ndarray,
#         metadata: List[Any],
#         batch_size: int = 100
#     ):
#         """Insert documents into ChromaDB"""
#         try:
#             # Process in batches
#             for i in range(0, len(texts), batch_size):
#                 end_idx = min(i + batch_size, len(texts))
#                 batch_texts = texts[i:end_idx]
#                 batch_embeddings = embeddings[i:end_idx]
                
#                 # Convert and sanitize metadata
#                 batch_metadata = [
#                     self._convert_metadata_to_dict(meta) 
#                     for meta in metadata[i:end_idx]
#                 ]
                
#                 # Generate IDs for the batch
#                 batch_ids = [f"doc_{j}" for j in range(i, end_idx)]
                
#                 # Add embeddings to collection
#                 self.collection.add(
#                     embeddings=batch_embeddings.tolist(),
#                     documents=batch_texts,
#                     metadatas=batch_metadata,
#                     ids=batch_ids
#                 )
            
#             self.logger.info(f"Successfully inserted {len(texts)} documents")
            
#         except Exception as e:
#             self.logger.error(f"Insert operation failed: {str(e)}")
#             raise

#     def hybrid_search(
#         self,
#         query_embedding: np.ndarray,
#         query_text: str,
#         limit: int = 5
#     ) -> List[Dict]:
#         """Perform hybrid search using both vector similarity and text matching"""
#         try:
#             # Query collection
#             results = self.collection.query(
#                 query_embeddings=query_embedding.reshape(1, -1).tolist(),
#                 n_results=limit,
#                 include=["documents", "metadatas", "distances"]
#             )
            
#             # Format results
#             formatted_results = []
#             if results['ids'] and len(results['ids'][0]) > 0:
#                 for i in range(len(results['ids'][0])):
#                     doc_id = results['ids'][0][i]
#                     score = 1 - float(results['distances'][0][i])  # Convert distance to similarity score
                    
#                     formatted_results.append({
#                         "id": doc_id,
#                         "text": results['documents'][0][i],
#                         "score": score,
#                         "metadata": results['metadatas'][0][i]
#                     })
            
#             return formatted_results
            
#         except Exception as e:
#             self.logger.error(f"Search operation failed: {str(e)}")
#             raise

#     def __del__(self):
#         """Cleanup resources"""
#         try:
#             if hasattr(self, 'client'):
#                 # ChromaDB PersistentClient doesn't need explicit closing
#                 pass
#         except Exception as e:
#             self.logger.warning(f"Error during cleanup: {str(e)}")



class ChromaDBVectorStore:
    """Handles vector storage and retrieval using ChromaDB"""
    
    def __init__(
        self,
        collection_name: str = "document_store",
        embedding_dim: int = 1024,
        persist_directory: str = None
    ):
        try:
            self.collection_name = collection_name
            self.embedding_dim = embedding_dim
            self.logger = logging.getLogger(__name__)
            
            # Use Streamlit's temp directory if no persist directory specified
            if persist_directory is None:
                persist_directory = os.path.join(tempfile.gettempdir(), "chromadb_data")
            
            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)
            
            
            # Initialize ChromaDB client with new configuration
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            self.collection = self._create_collection()
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def _create_collection(self):
        """Create ChromaDB collection if it doesn't exist"""
        try:
            # Delete collection if it exists
            try:
                self.client.delete_collection(self.collection_name)
            except Exception as e:
                self.logger.warning(f"No existing collection to delete: {str(e)}")
            
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
            
            return collection
            
        except Exception as e:
            self.logger.error(f"Failed to create collection: {str(e)}")
            raise

    def _sanitize_metadata_value(self, value: Any) -> Any:
        """Convert metadata values to ChromaDB-compatible types"""
        if value is None:
            return ""
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return str(list(value))
        elif isinstance(value, dict):
            return str(value)
        elif hasattr(value, '__dict__'):
            # Handle objects with __dict__ attribute
            return str(value.__dict__)
        else:
            return str(value)

    def _convert_metadata_to_dict(self, metadata: Any) -> Dict:
        """Convert metadata to ChromaDB-compatible dictionary format"""
        try:
            # If metadata is already a dict, use it as base
            if isinstance(metadata, dict):
                base_dict = metadata
            # If metadata has __dict__, use that
            elif hasattr(metadata, '__dict__'):
                base_dict = metadata.__dict__
            # If metadata is a ChunkMetadata object, extract its attributes
            elif isinstance(metadata, ChunkMetadata):
                base_dict = {
                    'source': metadata.source,
                    'chunk_index': metadata.chunk_index,
                    'total_chunks': metadata.total_chunks,
                    'start_char': metadata.start_char,
                    'end_char': metadata.end_char,
                    'word_count': metadata.word_count,
                    'page_number': metadata.page_number,
                    'section_title': metadata.section_title,
                    'semantic_density': metadata.semantic_density
                }
            else:
                base_dict = {"value": str(metadata)}

            # Sanitize all values in the dictionary
            return {
                k: self._sanitize_metadata_value(v)
                for k, v in base_dict.items()
            }
            
        except Exception as e:
            self.logger.warning(f"Error converting metadata to dict: {str(e)}")
            return {"error": "Failed to convert metadata"}

    def insert(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Any],
        batch_size: int = 100
    ):
        """Insert documents into ChromaDB"""
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                batch_texts = texts[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                
                # Convert and sanitize metadata
                batch_metadata = [
                    self._convert_metadata_to_dict(meta) 
                    for meta in metadata[i:end_idx]
                ]
                
                # Generate IDs for the batch
                batch_ids = [f"doc_{j}" for j in range(i, end_idx)]
                
                # Add embeddings to collection
                self.collection.add(
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
            
            self.logger.info(f"Successfully inserted {len(texts)} documents")
            
        except Exception as e:
            self.logger.error(f"Insert operation failed: {str(e)}")
            raise

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        limit: int = 5
    ) -> List[Dict]:
        """Perform hybrid search using both vector similarity and text matching"""
        try:
            # Query collection
            results = self.collection.query(
                query_embeddings=query_embedding.reshape(1, -1).tolist(),
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    score = 1 - float(results['distances'][0][i])  # Convert distance to similarity score
                    
                    formatted_results.append({
                        "id": doc_id,
                        "text": results['documents'][0][i],
                        "score": score,
                        "metadata": results['metadatas'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search operation failed: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'client'):
                # ChromaDB PersistentClient doesn't need explicit closing
                pass
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")






class HyDEGenerator:
    """Handles Hypothetical Document Embedding generation"""
    
    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def generate_hypothesis(
        self,
        query: str
    ) -> str:
        """Generate a hypothetical document for the query"""
        prompt = f"""Given the question: {query}
        Generate a detailed, hypothetical document that would perfectly answer this question.
        The document should be factual and informative."""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.5,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)





# class ReRanker:
#     """Handles reranking of retrieved documents"""
    
#     def __init__(
#         self,
#         # model_name: str = "BAAI/bge-reranker-base",
#         model_name: str = "BAAI/bge-reranker-v2-m3",
#         # model_name: str = "Alibaba-NLP/gte-multilingual-reranker-base",
#         device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     ):
#         self.device = device
#         logger.info(f"Initializing ReRanker on device: {self.device}")

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
#         self.model.eval()


#     def _prepare_pair(self, query: str, doc_text: str, max_length: int = 512) -> str:
#         """Prepare text pair for reranking"""
#         # Truncate text if needed while preserving query
#         query = query[:max_length // 4]  # Use 1/4 of max length for query
#         remaining_length = max_length - len(query) - 10  # Leave some space for special tokens
#         doc_text = doc_text[:remaining_length]
#         return query, doc_text


#     def rerank(
#         self,
#         query: str,
#         documents: List[Dict]
#     ) -> List[Dict]:
#         """Rerank documents using cross-encoder scoring"""
#         try:
#             logger.info(f"Reranking {len(documents)} documents")
#             reranked_docs = []
            
#             # # Process documents in batches
#             # batch_size = 1  # Small batch size to avoid memory issues
#             # for i in range(0, len(documents), batch_size):
#             #     batch_docs = documents[i:i + batch_size]
                
#              # Process each document in batch
#             for doc in documents:
#             # for doc in batch_docs:
#                 try:
#                     # Prepare input
#                     # doc_text = doc.get('text', '')
#                     doc_text = clean_text(doc.get('text', ''))
#                     truncated_query, truncated_text = self._prepare_pair(query, doc_text)
                    
#                     # Tokenize
#                     inputs = self.tokenizer(
#                         [truncated_query],
#                         [truncated_text],
#                         padding=True,
#                         truncation=True,
#                         max_length=512,
#                         return_tensors="pt"
#                     ).to(self.device)
                    
#                     # Get score
#                     with torch.no_grad():
#                         outputs = self.model(**inputs)
#                         # Get relevance score (using first logit as relevance score)
#                         score = torch.sigmoid(outputs.logits[0][0]).cpu().item()
                    
#                     # Combine with original score
#                     original_score = doc.get('score', 0.0)
#                     combined_score = 0.7 * score + 0.3 * original_score
                    
#                     doc_copy = doc.copy()
#                     doc_copy['rerank_score'] = float(combined_score)
#                     reranked_docs.append(doc_copy)
                    
#                 except Exception as e:
#                     logger.warning(f"Failed to rerank document: {str(e)}")
#                     # Keep original document with original score
#                     doc_copy = doc.copy()
#                     doc_copy['rerank_score'] = doc.get('score', 0.0)
#                     reranked_docs.append(doc_copy)
                
#                 # Clear CUDA cache if using GPU
#                 if self.device == "cuda":
#                     torch.cuda.empty_cache()
        
#             # Sort by rerank score
#             reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
#             logger.info("Reranking completed successfully")
            
#             return reranked_docs
        
            
#         except Exception as e:
#             logger.error(f"Reranking failed: {str(e)}")
#             # If reranking fails, return original documents with original scores
#             for doc in documents:
#                 return [dict(doc, rerank_score=doc.get('score', 0.0)) for doc in documents]
    
#         finally:
#             # Cleanup
#             if self.device == "cuda":
#                 torch.cuda.empty_cache()
    
#     def __del__(self):
#         """Cleanup resources"""
#         try:
#             if hasattr(self, 'device') and self.device == "cuda":
#                 torch.cuda.empty_cache()
#         except:
#             pass

        

class ReRanker:
    """Handles reranking of retrieved documents using Hugging Face pipeline"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        logger.info(f"Initializing ReRanker on device: {self.device}")

        # Initialize using pipeline
        try:
            from transformers import pipeline
            self.reranker = pipeline(
                "text-classification",
                model=model_name,
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {}
            )
        except Exception as e:
            logger.error(f"Failed to initialize reranker pipeline: {str(e)}")
            raise

    def _prepare_pair(self, query: str, doc_text: str, max_length: int = 512) -> str:
        """Prepare text pair for reranking"""
        query = query[:max_length // 4]
        remaining_length = max_length - len(query) - 10
        doc_text = doc_text[:remaining_length]
        return query, doc_text

    def rerank(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict]:
        """Rerank documents using pipeline scoring"""
        try:
            logger.info(f"Reranking {len(documents)} documents")
            reranked_docs = []
            
            for doc in documents:
                try:
                    # Clean and prepare text
                    doc_text = clean_text(doc.get('text', ''))
                    truncated_query, truncated_text = self._prepare_pair(query, doc_text)
                    
                    # Format input for reranker
                    pair = {"text": truncated_query, "text_pair": truncated_text}
                    
                    # Get score using pipeline
                    result = self.reranker(pair, truncation=True, max_length=512)
                    score = result[0]["score"]  # Get probability score
                    
                    # Combine with original score
                    original_score = doc.get('score', 0.0)
                    combined_score = 0.7 * score + 0.3 * original_score
                    
                    # Create reranked document
                    doc_copy = doc.copy()
                    doc_copy['rerank_score'] = float(combined_score)
                    reranked_docs.append(doc_copy)
                    
                except Exception as e:
                    logger.warning(f"Failed to rerank document: {str(e)}")
                    # Fallback to original score
                    doc_copy = doc.copy()
                    doc_copy['rerank_score'] = doc.get('score', 0.0)
                    reranked_docs.append(doc_copy)
                
                # Clean up GPU memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
            # Sort by rerank score
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            logger.info("Reranking completed successfully")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original documents with original scores
            return [dict(doc, rerank_score=doc.get('score', 0.0)) for doc in documents]
        
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'device') and self.device == "cuda":
                torch.cuda.empty_cache()
        except:
            pass






class DocumentSummarizer:
    """Handles both extractive and abstractive summarization of search results"""
    
    def __init__(
        self,
        model_name: str = "nijatzeynalov/mT5-based-azerbaijani-summarize"
    ):
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name
        )
    
    def summarize_results(
        self,
        results: List[Dict],
        query: str,
        language: str = "az"
    ) -> Dict[str, str]:
        """Generate both extractive and abstractive summaries"""
        try:
            # Generate both types of summaries
            # extractive_summary = self.extractive_summarize(results, language)
            abstractive_summary = self.abstractive_summarize(results, query, language)
            
            return {
                # 'extractive_summary': extractive_summary,
                'abstractive_summary': abstractive_summary
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return {
                # 'extractive_summary': self._get_default_message(language),
                'abstractive_summary': self._get_default_message(language)
            }
    
    def extractive_summarize(
        self,
        results: List[Dict],
        language: str
    ) -> str:
        """Create extractive summary by selecting key sentences"""
        try:
            if not results:
                return self._get_default_message(language)
            
            # Combine texts from top results, weighted by score
            sentences = []
            for result in results:
                text = clean_text(result.get('text', ''))
                score = float(result.get('score', 0.0))
                
                # Split into sentences
                result_sentences = text.split('.')
                for sentence in result_sentences:
                    sentence = sentence.strip()
                    if sentence:
                        sentences.append({
                            'text': sentence + '.',
                            'score': score
                        })
            
            # Sort sentences by score and select top ones
            sentences.sort(key=lambda x: x['score'], reverse=True)
            selected_sentences = sentences[:4]  # Select top 3 sentences
            
            # Combine selected sentences
            summary = ' '.join(sentence['text'] for sentence in selected_sentences)
            return clean_text(summary)
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {str(e)}")
            return self._get_default_message(language)
    
    def abstractive_summarize(
        self,
        results: List[Dict],
        query: str,
        language: str,
        max_length: int = 250,
        min_length: int = 80
    ) -> str:
        """Generate abstractive summary using language model"""
        try:
            if not results:
                return self._get_default_message(language)
            
            # Prepare input text with query context
            input_text = self._prepare_text_for_summary(results, query, language)
            
            # Generate summary
            summary = self.summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.4,
                top_p=0.9
            )[0]["summary_text"]
            
            return clean_text(summary)
            
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {str(e)}")
            return self._get_default_message(language)
    
    def _prepare_text_for_summary(
        self,
        results: List[Dict],
        query: str,
        language: str
    ) -> str:
        """Prepare text for summarization with query context"""
        
        prompt_template = {
            "az": f"""Sual: {query}

            Verilmiş suala əsasən və aşağıdakı məlumatlar əsasında qısa və dəqiq cavab hazırlayın:

            """,
            "en": f"""Question: {query}

            Based on the following information, prepare a brief and precise answer:

            """
        }
        
        text = prompt_template[language]
        
        # Add results, prioritizing higher scored ones
        # for result in sorted(results, key=lambda x: x.get('score', 0), reverse=True): 
        for result in sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:2]: # Top 2 
            result_text = clean_text(result.get('text', ''))
            if result_text:
                text += f"\n{result_text}"
        
        return text
    
    def _get_default_message(self, language: str) -> str:
        """Get default message when summarization fails"""
        return {
            "az": "Təəssüf ki, bu suala uyğun cavab tapılmadı.",
            "en": "Unfortunately, no relevant answer was found for this question."
        }[language]





class GPTDocumentSummarizer:
    """Handles document summarization using GPT"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini", 
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # self.client = OpenAI(api_key=api_key, http_client=httpx.Client(http2=True, verify=False)) if api_key else OpenAI()
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.logger = logging.getLogger(__name__)
    
    def summarize_results(
        self,
        results: List[Dict],
        repacked_data: str,
        query: str,
        language: str = "az"
    ) -> Dict[str, str]:
        """Generate both extractive and abstractive summaries"""
        try:
            # Find the index of 'Query' and 'Relevant Information' sections
            query_start = repacked_data.find('Query:')
            relevant_start = repacked_data.find('Relevant Information:')

            # Extract query and relevant information
            # query = repacked_data[query_start:relevant_start].strip() if not query else query
            query = repacked_data[query_start:relevant_start].strip().replace('\n', ' ')
            relevant_info = repacked_data[(relevant_start+len("Relevant Information: ")):].strip().replace('\n', ' ')

            # extractive_summary = self.extractive_summarize(results, relevant_info, language)
            abstractive_summary = self.abstractive_summarize(results, relevant_info, query, language)
            
            return {
                # 'extractive_summary': extractive_summary,
                'abstractive_summary': abstractive_summary,
                'relevant_info': relevant_info
            }
        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            return {
                # 'extractive_summary': self._get_default_message(language),
                'abstractive_summary': self._get_default_message(language),
                'relevant_info': ''
            }
    
    def extractive_summarize(
        self,
        results: List[Dict],
        relevant_info: str,
        language: str
    ) -> str:
        """Create extractive summary by selecting key sentences"""
        if not results:
            return self._get_default_message(language)
        
        # Combine texts from top results
        combined_text = "\n".join(r.get('text', '') for r in results[:3])
        
        system_prompts = {
            "az": """Siz təqdim edilən mətnlərdən verilmiş suala aid olan əsas faktları çıxararaq xülasə yaradan AI köməkçisisiniz.
            Orijinal mətnin ən vacib hissələrini verilmiş suala aid olacaq şəkildə seçib birləşdirin. Əgər verilmiş məlumatda suala aid cavab yoxdursa "Bağışlayın, bu haqda məlumatım yoxdur." şəklində cavab ver, heç bir halüsinasiya etmə.""",
            
            "en": """You are an AI assistant that creates summaries by extracting key facts from provided texts.
            Select and combine the most important parts of the original text."""
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompts[language]},
                        {"role": "user", "content": combined_text}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                self.logger.warning(f"Extractive summary attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return self._get_default_message(language)
    
    def abstractive_summarize(
        self,
        results: List[Dict],
        relevant_info: str,
        query: str,
        language: str
    ) -> str:
        """Generate abstractive summary using GPT"""
        if not results:
            return self._get_default_message(language)
        
        # Prepare context
        context = "\n".join(r.get('text', '') for r in results[:3])

        # Version 1:
        # system_prompts = {
        #     "az": """Siz sual və mətnlər əsasında qısa və aydın cavablar hazırlayan AI köməkçisisiniz.
        #     Cavablar konkret və əhatəli olmalıdır.""",
            
        #     "en": """You are an AI assistant that prepares brief and clear answers based on questions and texts.
        #     Answers should be specific and comprehensive."""
        # }
        
        # user_prompts = {
        #     "az": f"""Sual: {query}
            
        #     Məlumat:
        #     {relevant_info}
            
        #     Yuxarıdakı suala uyğun olaraq və verilən məlumatlar əsasında qısa və aydın cavab hazırlayın. Əgər verilmiş məlumatda suala aid heç bir informasiya yoxdursa "Bağışlayın, bu haqda məlumatım yoxdur." şəklində cavab ver, heç bir halüsinasiya etmə.""",
            
        #     "en": f"""Question: {query}
            
        #     Information:
        #     {relevant_info}
            
        #     Prepare a brief and clear answer based on the information above."""
        # }


        # Version 2:
        # system_prompts = {
        #     "az":"""Siz sual və mətnlər əsasında qısa və aydın cavablar hazırlayan AI köməkçisisiniz.
        #     Cavablar konkret və əhatəli olmalıdır.""",

        #     "en": """You are an AI assistant that prepares brief and clear answers based on questions and texts.
        #     Answers should be specific and comprehensive."""
        # }
        
        # user_prompts = {
        #     "az": f"""Sual: {query}
        
        #     Məlumat:
        #     {relevant_info}
            
        #     Yuxarıdakı suala uyğun olaraq və verilən məlumatlar əsasında qısa və aydın cavab hazırlayın. Sualın cavabı məlumatlarda varsa, ifadələr fərqli olsa belə, uyğun cavab verin. Əgər verilmiş məlumatda suala aid heç bir informasiya yoxdursa, "Bağışlayın, bu haqda məlumatım yoxdur." şəklində cavab ver. Cavabı yalnız mətn şəklində təqdim edin, əlavə etiketlər və ya sualı daxil etməyin. Heç bir halüsinasiya etmə.

        #     **Misallar:**

        #     1. **Sual:** "İşəmuzd əmək haqqı nədir?"
        #     - **Cavab:** "İşəmuzd əmək haqqı, işçilərin müəyyən olunmuş dərəcələr üzrə yerinə yetirdiyi işin həcminə görə hesablanan aylıq əmək haqqıdır."

        #     2. **Sual:** "İşçilərin yerinə yetirdiyi işin həcminə əsasən aylıq maaş necə təyin olunur?"
        #     - **Cavab:** "İşəmuzd əmək haqqı, işçilərin gördüyü işin həcminə və müəyyən olunmuş dərəcələrə görə hesablanır."

        #     3. **Sual:** "Bank işçilərinin illik məzuniyyət hüquqları nə qədərdir?"
        #     - **Cavab:** "Bağışlayın, bu haqda məlumatım yoxdur."

        #     Bu nümunələr əsasında, sualın cavabını tapmağa çalış və yuxarıdakı qaydalara əməl et."""
        # }



        # Version 3:
        # system_prompts = {
        #     "az": """Siz sual və mətnlər əsasında qısa və aydın cavablar hazırlayan AI köməkçisisiniz.
        #     Cavablar konkret və əhatəli olmalıdır.
            
        #     Əgər verilmiş məlumatda sualın birbaşa cavabı yoxdursa, amma əlaqəli məlumat varsa, 
        #     həmin əlaqəli məlumatı təqdim edin və bunun dolayı cavab olduğunu bildirin.""",

        #     "en": """You are an AI assistant that prepares brief and clear answers based on questions and texts.
        #     Answers should be specific and comprehensive.
            
        #     If the direct answer is not in the provided information but related information exists,
        #     provide that related information and indicate it's an indirect response."""
        # }

        # Version 3.1 (enhancing prompt) 
        # system_prompts = {
        #     "az": """Siz sual və mətnlər əsasında qısa və aydın cavablar hazırlayan AI köməkçisisiniz.

        #     Əsas qaydalar:
        #     1. Cavablar konkret və əhatəli olmalıdır
        #     2. Yalnız verilmiş məlumatlardan istifadə edin
        #     3. Məlumatda olmayan heç bir şey əlavə etməyin
        #     4. Cavabı formalaşdırarkən aşağıdakı addımları izləyin:
        #     - Əvvəlcə verilmiş məlumatları analiz edin
        #     - Sualın birbaşa cavabını axtarın
        #     - Birbaşa cavab yoxdursa, əlaqəli məlumatları axtarın
        #     - Heç bir əlaqəli məlumat yoxdursa, bunu bildirin

        #     Cavab formatı:
        #     - Qısa və konkret olmalı
        #     - 1-3 cümlədən ibarət olmalı
        #     - Sadə dildə izah edilməli""",

        #     "en": """You are an AI assistant that prepares brief and clear answers based on questions and provided texts.

        #     Main rules:
        #     1. Answers must be specific and comprehensive
        #     2. Only use information from the provided text
        #     3. Don't add any information not present in the source
        #     4. Follow these steps when forming an answer:
        #     - First analyze the provided information
        #     - Look for direct answers to the question
        #     - If no direct answer, look for related information
        #     - If no related information exists, state this

        #     Response format:
        #     - Keep it brief and specific
        #     - Use 1-3 sentences
        #     - Explain in simple terms
        #     - Minimize use of technical terms"""
        # }

        # user_prompts = {
        #     "az": f"""Sual: {query}

        #     Məlumat:
        #     {relevant_info}
            
        #     Qaydalar:
        #     1. Verilən məlumatlar əsasında qısa və aydın cavab hazırlayın
        #     2. Sualın birbaşa cavabı məlumatlarda varsa, onu təqdim edin
        #     3. Birbaşa cavab yoxdursa amma əlaqəli məlumat varsa, onu təqdim edin
        #     4. Heç bir əlaqəli məlumat yoxdursa, "Bağışlayın, bu haqda məlumatım yoxdur" yazın
        #     5. Cavabı yalnız mətn formasında verin
        #     6. Əlavə etiketlər və ya sualı təkrarlamayın
        #     7. Xəyal məhsulu (hallüsinasiya) məlumatlar əlavə etməyin""",

        #     "en": f"""Question: {query}

        #     Information:
        #     {relevant_info}
            
        #     Rules:
        #     1. Prepare a brief and clear answer based on the provided information
        #     2. If the direct answer exists in the information, provide it
        #     3. If no direct answer exists but related information is available, provide that
        #     4. If no related information exists, respond with "Sorry, I don't have information about this"
        #     5. Provide the answer in text format only
        #     6. Don't include additional tags or repeat the question
        #     7. Don't add any hallucinated information"""
        # }

        # Version 4 (for no answer decreasing with temp 0,4):
        # system_prompts = {
        #     "az": """Siz verilən sual əsasında mətnlərdən uyğun cavab hazırlayan AI köməkçisisiniz.

        #     Əsas qaydalar:
        #     1. Məlumatları geniş perspektivdə analiz edin
        #     2. Birbaşa cavabları üstün tutun, lakin dolayı əlaqəli məlumatları da nəzərə alın:
        #     - Oxşar konsepsiyalar və anlayışlar
        #     - Eyni mövzu daxilində əlaqəli fikirlər
        #     - Məntiqi əlaqələr və nəticələr
        #     3. "Məlumat yoxdur" cavabını yalnız aşağıdakı hallarda verin:
        #     - Heç bir birbaşa və ya dolayı əlaqə yoxdursa
        #     - Məlumatlar tamamilə fərqli mövzudadırsa
        #     4. Qeyri-müəyyənlik olduqda "məlumat yoxdur" əvəzinə mövcud əlaqəli məlumatı təqdim edin""",

        #     "en": """You are an AI assistant that prepares precise answers based on questions and texts.

        #     Core rules:
        #     1. Analyze information from a broad perspective
        #     2. Prefer direct answers but also consider indirect relationships:
        #     - Similar concepts and ideas
        #     - Related points within the same topic
        #     - Logical connections and implications
        #     3. Only respond "no information" when:
        #     - No direct or indirect relationships exist
        #     - Information is completely unrelated to the topic
        #     4. When uncertain, provide related information instead of "no information"
            
        #     Important: When in doubt between saying "no information" or providing a related answer,
        #     ALWAYS choose to provide the related answer with appropriate context."""
        # }

        # user_prompts = {
        #     "az": f"""Sual: {query}

        #     Məlumat:
        #     {relevant_info}
            
        #     Təlimatlar:
        #     1. Əvvəlcə birbaşa cavab axtarın
        #     2. Birbaşa cavab yoxdursa, dolayı əlaqəli məlumatları analiz edin:
        #     - Eyni mövzuda olan məlumatlar
        #     - Oxşar konsepsiyalar
        #     - Məntiqi əlaqələr
        #     3. Yalnız heç bir əlaqə tapılmadıqda "məlumat yoxdur" deyin
        #     4. Qeyri-müəyyənlik halında əlaqəli məlumatı təqdim edin""",

        #     "en": f"""Question: {query}

        #     Information:
        #     {relevant_info}
            
        #     Instructions:
        #     1. First look for direct answers
        #     2. If no direct answer, analyze related information:
        #     - Content in the same topic area
        #     - Similar concepts
        #     - Logical connections
        #     3. Only say "no information" if truly nothing is related
        #     4. When uncertain, provide related information with context"""
        # }


        # Version 5 (fixing etiket issue, user-friendly answers) 
        # system_prompts = {
        #     "az": """Siz verilən sual əsasında mətnlərdən uyğun və istifadəçi dostu cavab hazırlayan AI köməkçisisiniz.

        #     Əsas qaydalar:
        #     1. Məlumatları geniş perspektivdə analiz edin
        #     2. Birbaşa cavabları üstün tutun, lakin dolayı əlaqəli məlumatları da nəzərə alın:
        #         - Oxşar konsepsiyalar və anlayışlar
        #         - Eyni mövzu daxilində əlaqəli fikirlər
        #         - Məntiqi əlaqələr və nəticələr
        #     3. "Məlumat yoxdur" cavabını yalnız aşağıdakı hallarda verin:
        #         - Heç bir birbaşa və ya dolayı əlaqə yoxdursa
        #         - Məlumatlar tamamilə fərqli mövzudadırsa
        #     4. Qeyri-müəyyənlik olduqda "məlumat yoxdur" əvəzinə mövcud əlaqəli məlumatı təqdim edin

        #     Təqdimat qaydaları:
        #     1. Cavabları istifadəçi dostu, təbii danışıq dilində, struktural etiketlər olmadan təqdim edin
        #     2. "Məlumatda..." əvəzinə "Məndə olan məlumata əsasən..." və ya "Mövcud məlumata görə..." kimi ifadələr istifadə edin
        #     3. Bütün cavabları vahid, axıcı mətn formasında yazın""",

        #     "en": """You are an AI assistant that prepares precise and user-friendly answers based on questions and texts.

        #     Core rules:
        #     1. Analyze information from a broad perspective
        #     2. Prefer direct answers but also consider indirect relationships:
        #         - Similar concepts and ideas
        #         - Related points within the same topic
        #         - Logical connections and implications
        #     3. Only respond "no information" when:
        #         - No direct or indirect relationships exist
        #         - Information is completely unrelated to the topic
        #     4. When uncertain, provide related information instead of "no information"

        #     Presentation rules:
        #     1. Present answers in user-friendly, natural conversational language without structural labels
        #     2. Use phrases like "Based on the available information..." instead of "In the information..."
        #     3. Write all responses as unified, flowing text"""
        # }

        # user_prompts = {
        #     "az": f"""Sual: {query}

        #     Məlumat:
        #     {relevant_info}
            
        #     Təlimatlar:
        #     1. Əvvəlcə birbaşa cavab axtarın
        #     2. Birbaşa cavab yoxdursa, dolayı əlaqəli məlumatları analiz edin:
        #         - Eyni mövzuda olan məlumatlar
        #         - Oxşar konsepsiyalar
        #         - Məntiqi əlaqələr
        #     3. Yalnız heç bir birbaşa və ya dolayı əlaqə tapılmadıqda "məlumat yoxdur" deyin
        #     4. Qeyri-müəyyənlik halında əlaqəli məlumatı təqdim edin
            
        #     Təqdimat formatı:
        #     1. Cavabı istifadəçi dostu, təbii və axıcı dildə hazırlayın
        #     2. Struktur etiketləri istifadə etməyin
        #     3. "Məlumatda..." əvəzinə "Məndə olan məlumata əsasən..." və ya "Mövcud məlumata görə..." kimi ifadələr işlədin""",

        #     "en": f"""Question: {query}

        #     Information:
        #     {relevant_info}
            
        #     Instructions:
        #     1. First look for direct answers
        #     2. If no direct answer, analyze related information:
        #         - Content in the same topic area
        #         - Similar concepts
        #         - Logical connections
        #     3. Only say "no information" if truly nothing is related
        #     4. When uncertain, provide related information with context

        #     Presentation format:
        #     1. Prepare the answer in user-friendly, natural flowing language
        #     2. Avoid structural labels
        #     3. Use phrases like "Based on the available information..." instead of "In the information..." """
        # }


        # version 5 with upgraded(with dont know cases...)
        system_prompts = {
            "az": """Siz verilən sual əsasında mətnlərdən uyğun və dəqiq cavab hazırlayan AI köməkçisisiniz.

            Əsas qaydalar:
            1. Məlumatları hərtərəfli analiz edin:
                - Birbaşa cavabları əsas götürün
                - Sualın mahiyyətini və məqsədini anlayın
                - Müxtəlif ifadə formalarını tanıyın (məsələn, "əmək haqqı" və "maaş")
                - Məna baxımından oxşar anlayışları nəzərə alın və əlaqəli məlumatları tapın
            2. Cavab vermə qaydaları:
                - Yalnız dəqiq uyğun gələn və ya əlaqəli məlumatları istifadə edin
                - Uzaq əlaqələr qurmaqdan çəkinin, amma yaxın və əlaqəli məlumatları istifadə edərək cavab verin
                - Təxmin və fərziyyələr irəli sürməyin
            3. "Məlumat tapılmadı" cavabını verin:
                - Verilmiş məlumatlarda birbaşa və ya dolayı cavab olmadıqda
                - Sual tamamilə fərqli mövzu və ya sahəyə aid olduqda
            4. **Kontekstdəki əlaqəli məlumatları istifadə edin**:
                - Məlumat tapılmadıqda belə, kontekstdəki mövcud məlumatlardan istifadə edərək cavab verin.
            
            Təqdimat qaydaları:
            1. Cavabları təbii və anlaşılan dildə verin
            3. Cavabları bir mətn şəklində, ardıcıl formada yazın
            """
        }

        user_prompts = {
            "az": f"""Sual: {query}

            Məlumat:
            {relevant_info}
            
            Təlimatlar:
            1. Məlumatları diqqətlə analiz edin:
                - Birbaşa uyğun cavabları axtarın
                - Sualın məna baxımından oxşar ifadələrini və bağlı anlayışları nəzərə alın
                - Fərqli ifadə formalarını (məsələn, "əmək haqqı" və "maaş") tanıyın
            2. Əgər cavab tapılmazsa, "Hörmətli istifadəçi, bağışlayın, bu suala uyğun məlumat tapılmadı" deyin
            3. Kontekstdəki əlaqəli məlumatları istifadə edin və buna əsaslanaraq cavab verin.
            """
        }





        # Version 5.1 (prompt concise ..) # check it
        # system_prompts = {
        #     "az": """Siz istifadəçi dostu və dəqiq cavablar hazırlayan AI köməkçisisiniz.

        #     Analiz qaydaları:
        #     1. Məlumatları hərtərəfli analiz edin:
        #         - Birbaşa uyğun cavabları prioritetləşdirin
        #         - Dolayı əlaqəli məlumatları nəzərə alın (oxşar konsepsiyalar, əlaqəli fikirlər)
        #         - Məntiqi əlaqələr və nəticələr çıxarın
        #     2. Cavab yoxdursa:
        #         - Əlaqəli məlumatları təqdim edin
        #         - Yalnız heç bir əlaqə olmadıqda "məlumat yoxdur" deyin

        #     Təqdimat qaydaları:
        #     1. İstifadəçi dostu və təbii dildə yazın:
        #         - Struktur etiketləri ("Birbaşa cavab:", "Əlaqəli məlumat:" kimi) istifadə etməyin
        #         - "Məlumatda..." əvəzinə "Məndə olan məlumata əsasən..." və ya "Mövcud məlumata görə..." işlədin
        #         - Bütün cavabı vahid, axıcı mətn kimi tərtib edin""",

        #     "en": """You are an AI assistant that prepares user-friendly and precise answers.

        #     Analysis rules:
        #     1. Analyze information comprehensively:
        #         - Prioritize direct relevant answers
        #         - Consider indirect relationships (similar concepts, related points)
        #         - Draw logical connections and conclusions
        #     2. When no direct answer exists:
        #         - Provide related information
        #         - Only say "no information" if nothing is related

        #     Presentation rules:
        #     1. Write in user-friendly and natural language:
        #         - Avoid structural labels ("Direct answer:", "Related information:" etc.)
        #         - Use "Based on available information..." instead of "In the information..."
        #         - Compose the entire response as unified, flowing text"""
        # }

        # user_prompts = {
        #     "az": f"""Sual: {query}

        #     Məlumat:
        #     {relevant_info}
            
        #     Cavabı hazırlayarkən:
        #     1. Məlumatı hərtərəfli analiz edin və ən uyğun cavabı tapın
        #     2. İstifadəçi dostu və təbii dildə izah edin
        #     3. "Məndə olan məlumata əsasən..." və ya "Mövcud məlumata görə..." kimi təbii ifadələr işlədin
        #     4. Bütün cavabı vahid, rahat oxunan mətn şəklində təqdim edin""",

        #     "en": f"""Question: {query}

        #     Information:
        #     {relevant_info}
            
        #     When preparing the answer:
        #     1. Analyze information thoroughly and find the most relevant answer
        #     2. Explain in user-friendly and natural language
        #     3. Use natural phrases like "Based on available information..."
        #     4. Present the entire response as unified, easily readable text"""
        # }


        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompts[language]},
                        {"role": "user", "content": user_prompts[language]}
                    ],
                    # temperature=0.7,
                    temperature=0.4, # version 4
                    max_tokens=500
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                self.logger.warning(f"Abstractive summary attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return self._get_default_message(language)
    
    def _get_default_message(self, language: str) -> str:
        """Get default message when summarization fails"""
        return {
            "az": "Təəssüf ki, bu suala uyğun cavab tapılmadı.",
            "en": "Unfortunately, no relevant answer was found for this question."
        }[language]








class QuestionGenerator:
    """Generates relevant questions based on document content"""
    
    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        device: str = None
    ):
        # Set environment variable to handle forking warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_questions(
        self,
        text: str,
        num_questions: int = 3,
        language: str = "az"
    ) -> List[str]:
        """Generate questions based on the document content"""
        
        system_prompt = {
            "az": "Siz verilmiş sualdan oxşar və əlaqəli suallar yaradan AI köməkçisisiniz. Təqdim olunan sualın əsas mövzularına uyğun sadə və aydın suallar yaradın.",
            "en": "You are an AI assistant that generates similar and relevant questions based on the given question. Create simple and clear questions that cover the main topics of the provided question."
        }[language]
        
        user_prompt = {
            "az": f"""
            Aşağıdakı sualdan 5 oxşar sual yaradın.
            Suallar mətnin əsas mövzularını əhatə etməli və sual işarəsi ilə bitməlidir.
            Yalnız sualları qaytarın, heç bir əlavə mətn olmadan.
            
            Sual:
            {text}
            """,
            "en": f"""
            Create 5 similar questions based on the question below.
            Questions should cover the main topics and end with a question mark.
            Return only the questions, no additional text.
            
            Question:
            {text}
            """
        }[language]
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, #
                repetition_penalty=1.2 # 
            )
        
        # Process output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract questions
        questions = []
        for line in generated_text.split('\n'):
            line = line.strip()
            # Only keep lines that look like questions
            if line and '?' in line and not line.startswith(('Mətn:', 'Text:')):
                questions.append(line)

        # Ensure we don't have duplicates
        questions = list(dict.fromkeys(questions))
        
        # Add original query if not enough variations
        if text not in questions:
            questions.insert(0, text)
            
        print("\nGenerated Questions: ", questions)            
        return questions[:num_questions]




# old but worked version
# class GPTQuestionGenerator:
#     """Generates relevant questions using GPT model"""
    
#     def __init__(
#         self,
#         api_key: str = None,
#         model: str = "gpt-4o-mini",
#         max_retries: int = 3,
#         retry_delay: int = 1
#     ):
#         self.model = model
#         self.max_retries = max_retries
#         self.retry_delay = retry_delay
        
#         # Initialize OpenAI client
#         self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
#         self.logger = logging.getLogger(__name__)

#     def generate_questions(
#         self,
#         text: str,
#         num_questions: int = 2,
#         language: str = "az"
#     ) -> List[str]:
#         """Generate questions based on the document content"""
        
#         system_prompts = {
#             "az": """Siz verilmiş sualdan oxşar və əlaqəli suallar yaradan AI köməkçisisiniz.
#             Sadə və aydın suallar yaradın. Hər sual sual işarəsi ilə bitməlidir.""",
            
#             "en": """You are an AI assistant that generates similar and related questions from a given question.
#             Create simple and clear questions. Each question should end with a question mark."""
#         }

#         user_prompts = {
#             "az": f"""Bu sualdan eyni məzmunlu {num_questions} oxşar sual yaradın:
#             {text}
            
#             Qaydalar:
#             - Suallar qısa və konkret olmalıdır
#             - Hər sual sual işarəsi ilə bitməlidir
#             - Yalnız sualları qaytarın, əlavə mətn yazmayın
#             - Hər sual yeni sətirdən başlamalıdır""",
            
#             "en": f"""Create {num_questions} similar questions from this question:
#             {text}
            
#             Rules:
#             - Questions should be short and specific
#             - Each question must end with a question mark
#             - Return only the questions, no additional text
#             - Each question should start on a new line"""
#         }


#         self.max_retries = 1 ## check it 

#         for attempt in range(self.max_retries):
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": system_prompts[language]},
#                         {"role": "user", "content": user_prompts[language]}
#                     ],
#                     temperature=0.4,
#                     max_tokens=500
#                 )
                
#                 # Extract questions from response
#                 generated_text = response.choices[0].message.content
#                 questions = [q.strip() for q in generated_text.split('\n') if '?' in q]
                
#                 # Add original question if not in list
#                 if text not in questions:
#                     questions.insert(0, text)
                
#                 return questions[:num_questions]
                
#             except Exception as e:
#                 self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(self.retry_delay)
#                 else:
#                     self.logger.error("Failed to generate questions")
#                     return [text]  # Return original question if all attempts fail






# new with extra fixing to azerbaijani word, return if is not related ..
class GPTQuestionGenerator:
    """Generates relevant questions using GPT model"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize OpenAI client
        # self.client = OpenAI(api_key=api_key, http_client=httpx.Client(http2=True, verify=False)) if api_key else OpenAI()
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.logger = logging.getLogger(__name__)


    # Version 1 (one prompt): 
    # def generate_questions(
    #     self,
    #     text: str,
    #     num_questions: int = 2,
    #     language: str = "az"
    # ) -> List[str]:
    #     """Generate questions based on the document content"""
        
    #     system_prompts = {
    #         # "az": """Siz verilmiş sualdan oxşar və əlaqəli suallar yaradan AI köməkçisisiniz.
    #         # Sadə və aydın suallar yaradın. Hər sual sual işarəsi ilə bitməlidir.""",

    #         "az": "Siz bank sahəsində Insan Resursları üzrə ixtisaslaşmış süni intellektsiz (AI köməkçisiniz). Sizin vəzifəniz verilmiş sualı düzəldib ondan eyni məzmunlu suallar yaratmaqdır. Sadə və aydın suallar yaradın. Hər sual sual işarəsi ilə bitməlidir.",
            
    #         "en": """You are an AI assistant that generates similar and related questions from a given question.
    #         Create simple and clear questions. Each question should end with a question mark."""
    #     }

    #     user_prompts = {
    #         # "az": f"""Bu sualdan eyni məzmunlu {num_questions} oxşar sual yaradın:
    #         # {text}
            
    #         # Qaydalar:
    #         # - Suallar qısa və konkret olmalıdır
    #         # - Hər sual sual işarəsi ilə bitməlidir
    #         # - Yalnız sualları qaytarın, əlavə mətn yazmayın
    #         # - Hər sual yeni sətirdən başlamalıdır""",
            
    #         "az": f"""
    #         Sizin vəzifəniz qarışıq və ya tamamlanmamış bir mətni düzgünləşdirmək və Azərbaycan dilində bank sektoru üzrə insan resursları sahəsinə uyğun şəkildə təqdim etməkdir. Verilən mətnin qeyri-kamil və ya qarışıq formatda olduğunu nəzərə alaraq, onu düzgün şəkildə təkmilləşdirin.

    #         Qeyd:
    #         1. **Mətn azərbaycan dilində tam düzgün yazılıbsa** o zaman verilən sualın özünü heç bir dəyişiklik etmədən qaytar.
    #         2. **Azərbaycan dilindəki sözlər** (məsələn, "telimler" yerinə "təlimlər") **latın əlifbası ilə** yazılıbsa, onları düzgün Azərbaycan yazılışına çevirin.
    #         3. **Başqa dillərdən** (ingilis, rus və s.) olan sözləri tapdıqda, onları **Azərbaycan dilindəki qarşılığı ilə** əvəz edin (məsələn, "training" -> "təlim", "employee" -> "işçi", "program" -> "proqram").
    #         4. **Əgər mətndə başqa dildən olan bir sözün Azərbaycan dilində birbaşa qarşılığı tapılmırsa**, o zaman **kontekstə uyğun düzgün Azərbaycan sözünü seçin**. Məsələn, əgər "training" sözü bir işçi təliminə aiddirsə, onu "təlim" olaraq düzəldin. Hər halda, **məqsəd doğru mənanı və mətni qorumaqdır**.

    #         Aşağıda olan nümunə düzəlişlərə baxın:
    #         - "Əsas fəaliyyət göstəriciləri nədir?" -> "Əsas fəaliyyət göstəriciləri nədir?"
    #         - "Telim programları" -> "Təlim proqramları"
    #         - "Employee benefits" -> "İşçi faydaları"
    #         - "Kapital bankada ingilzce telimler olurmu?" -> "Kapital Bankda ingilis dilində təlimlər varmı?"
    #         - "Kapital Bankda ingilzce programalar olacakmi?" -> "Kapital Bankda ingilis dilində proqramlar olacaq?"

    #         Burada olan mətni yalnızca düzəldin və düzgün şəkildə təqdim edin. Məqsədimiz, bank sektorunda insan resursları və istedadın idarə olunması ilə əlaqəli mətni düzgünləşdirməkdir.

    #         Həmçinin, düzəldilmiş mətni əsas alaraq, eyni məzmunlu {num_questions} oxşar sual yaradın:
            
    #         Burada olan mətn:
    #         {text}
            
    #         Qaydalar:
    #         - İlk sual olaraq mütləq eyni verilən sualın azərbaycan dilində yazılmış düzgün formasını qaytar. Əgər verilmiş sual azərbaycan dilindədirsə o zaman verilən sualın özünü heç bir dəyişiklik etmədən qaytar.
    #         - Suallar qısa və konkret olmalıdır
    #         - Hər sual sual işarəsi ilə bitməlidir
    #         - Yalnız sualları qaytarın, əlavə mətn yazmayın
    #         - Hər sual yeni sətirdən başlamalıdır

    #         Qeyd: Əgər bir sualda sual işarəsi yoxdursa, amma cümlə sual formatında(məntiqində) yazılıbsa, məsələn, "İşdə istirahət etmək olar" kimi, onu düzgün tərcümə edərək sual halına gətirin. Bu tip hallarda belə cümlələr hələ də sual olaraq qiymətləndirilməlidir. Məsələn, "İşdə istirahət etmək olar?" şəklində düzəldin.
    #         """,
    #         # Əgər suallar insan resursları mövzusu ilə əlaqəli deyilsə, "Bağışlayın, suallar insan resursları mövzusu ilə əlaqəli deyil." mesajını qaytarın.


    #         "en": f"""Create {num_questions} similar questions from this question:
    #         {text}
            
    #         Rules:
    #         - Questions should be short and specific
    #         - Each question must end with a question mark
    #         - Return only the questions, no additional text
    #         - Each question should start on a new line"""
    #     }


    #     self.max_retries = 1 ## check it 

    #     for attempt in range(self.max_retries):
    #         try:
    #             response = self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=[
    #                     {"role": "system", "content": system_prompts[language]},
    #                     {"role": "user", "content": user_prompts[language]}
    #                 ],
    #                 # temperature=0.4,
    #                 temperature=0.2,
    #                 max_tokens=500
    #             )
                
    #             # Extract questions from response
    #             generated_text = response.choices[0].message.content

    #             if "Bağışlayın, suallar insan resursları mövzusu ilə əlaqəli deyil." in generated_text:
    #                 return '' 
    #                 # return [] 

    #             questions = [q.strip() for q in generated_text.split('\n') if '?' in q]
                
    #             # Add original question if not in list
    #             if text in questions:
    #                 questions.remove(text)
                

    #             return questions[:num_questions]
                
    #         except Exception as e:
    #             self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
    #             if attempt < self.max_retries - 1:
    #                 time.sleep(self.retry_delay)
    #             else:
    #                 self.logger.error("Failed to generate questions")
    #                 return [text]  # Return original question if all attempts fail


    # Version 2 (separate prompts): 
    def _fix_azerbaijani_text(self, text: str) -> str:
        """Fix and improve Azerbaijani text formatting and language"""
        
        system_prompt = """Siz bank sahəsində Insan Resursları üzrə ixtisaslaşmış süni intellektsiz (AI köməkçisiniz). 
        Sizin vəzifəniz verilmiş sualı düzəldib təkmilləşdirməkdir."""
        
        user_prompt = f"""
        Verilmiş mətni düzgünləşdirməli və Azərbaycan dilində bank sektoru üzrə insan resursları sahəsinə uyğun şəkildə təqdim etməlisiniz.

        Qeyd:
        1. **Mətn azərbaycan dilində tam düzgün yazılıbsa** heç bir dəyişiklik etməyin.
        2. **Azərbaycan dilindəki sözlər** latın əlifbası ilə yazılıbsa (məsələn, "telimler" yerinə "təlimlər"), onları düzgün Azərbaycan yazılışına çevirin.
        3. **Başqa dillərdən** olan sözləri Azərbaycan dilindəki qarşılığı ilə əvəz edin (məsələn, "training" -> "təlim").
        4. **Əgər sual məntiqi və ya qrammatik səhvlər varsa**, onları düzəldin.
        5. Sual işarəsi yoxdursa, amma cümlə sual formatındadırsa, sual işarəsi əlavə edin.

        Verilmiş mətn:
        {text}

        Yalnız düzəldilmiş mətni qaytarın, əlavə izahat yazmayın."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1  # Lower temperature for more consistent corrections
            )
            
            fixed_text = response.choices[0].message.content.strip()
            return fixed_text
            
        except Exception as e:
            self.logger.warning(f"Text fixing failed: {str(e)}")
            return text

    def _generate_variations(self, fixed_query: str, num_variations: int) -> List[str]:
        """Generate variations of the fixed query"""
        
        system_prompt = """Siz bank sahəsində Insan Resursları üzrə ixtisaslaşmış süni intellektsiz (AI köməkçisiniz). 
        Siz verilmiş sualdan eyni məzmunlu yeni suallar yaradırsınız."""

        user_prompt = f"""
        Bu sualdan eyni məzmunlu {num_variations} oxşar sual yaradın:
        {fixed_query}
        
        Qaydalar:
        - Suallar qısa və konkret olmalıdır
        - Hər sual sual işarəsi ilə bitməlidir
        - Yalnız sualları qaytarın, əlavə mətn yazmayın
        - Hər sual yeni sətirdən başlamalıdır
        - Verilən orijinal sualı təkrarlamayın
        
        Əgər suallar insan resursları mövzusu ilə əlaqəli deyilsə, "Bağışlayın, suallar insan resursları mövzusu ilə əlaqəli deyil." mesajını qaytarın."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            if "Bağışlayın, suallar insan resursları mövzusu ilə əlaqəli deyil." in generated_text:
                return []
                
            variations = [q.strip() for q in generated_text.split('\n') if '?' in q]
            return variations[:num_variations]
            
        except Exception as e:
            self.logger.warning(f"Variation generation failed: {str(e)}")
            return []

    def generate_questions(
        self,
        text: str,
        num_questions: int = 2,
        language: str = "az"
    ) -> List[str]:
        """Generate questions based on the input text with fixing and variations"""
        try:
            # Step 1: Fix and improve the input text
            fixed_query = self._fix_azerbaijani_text(text) if language == "az" else text
            
                
            # Step 2: Generate variations
            # variations = self._generate_variations(fixed_query, num_questions)
            
            # Combine results
            all_questions = [fixed_query]  # Start with the fixed original query
            # all_questions.extend(variations)  # Add variations
            
            # Remove duplicates while preserving order
            seen = set()
            unique_questions = []
            for q in all_questions:
                if q not in seen:
                    seen.add(q)
                    unique_questions.append(q)
                    
            return unique_questions[:num_questions + 1]  # Include original + variations
            
        except Exception as e:
            self.logger.error(f"Question generation failed: {str(e)}")
            return [text] if text else ''








class ResourceManager:
    """Manages cleanup of system resources and multiprocessing objects"""
    
    _initialized = False
    _semaphores = set()
    
    @classmethod
    def initialize(cls):
        """Initialize resource manager"""
        if not cls._initialized:
            # Set multiprocessing start method
            if current_process().name == 'MainProcess':
                try:
                    mp.set_start_method('spawn', force=True)
                except RuntimeError:
                    pass
            
            # Register cleanup handler
            atexit.register(cls.cleanup)
            cls._initialized = True
    
    @classmethod
    def cleanup(cls):
        """Cleanup all resources"""
        # Clean up CUDA resources
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Clean up Milvus connection
        try:
            from pymilvus import connections
            connections.disconnect("default")
        except:
            pass
        
        # Clean up multiprocessing resources
        try:
            # Clean up any remaining semaphores
            for sem in cls._semaphores:
                try:
                    sem.unlink()
                except:
                    pass
            cls._semaphores.clear()
            
            # Clean up any remaining processes
            if current_process().name == 'MainProcess':
                for p in mp.active_children():
                    p.terminate()
        except:
            pass






class RAGPipeline:
    """Main RAG Pipeline class that orchestrates all components"""
    
#     def __init__(
#         self,
#         collection_name: str = "azerbaijan_docs",
#         chunk_size: int = 512, # 1024
#         chunk_overlap: int = 50, # 128
#         # embedding_model: str = "BAAI/bge-m3", # Local model
#         # embedding_model: str = "LocalDoc/TEmA-small", # Local model
#         embedding_model="text-embedding-3-small",  # OpenAI model
#         host: str = "localhost",
#         port: int = 19530,
#         language: str = "az",
#         device: str = None,
#         batch_size: int = 4,
#         # cache_dir: str = "embeddings_cache",  # New parameter
#         cache_dir: str = "cache"  # Add this parameter

#         ):
        
#         """Initialize the RAG Pipeline with all components"""
        
#         # atexit.register(ResourceManager.cleanup)
#         ResourceManager.initialize()

#         # Set environment variables to suppress warnings
#         os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:resource_tracker'
#         # Set memory constraints
#         os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'

#        # Initialize components with proper resource handling
#         if torch.cuda.is_available():
#             # Use spawn method for CUDA tensors
#             torch_mp.set_start_method('spawn', force=True)



#         logger.info("Initializing pipeline components...")

#         try: 
#             """Initialize pipeline with memory optimization"""

#             self.logger = logging.getLogger(__name__)

#             self.batch_size = batch_size
#             self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

#             self.language = language
            
# # cache
#             self.cache_dir = Path(cache_dir)
#             self.cache_dir.mkdir(parents=True, exist_ok=True)
# # 
    
#             # Initialize components
#             self.doc_processor = DocumentProcessor()
#             self.chunking_strategy = ChunkingStrategy(
#                 chunk_size=chunk_size,
#                 chunk_overlap=chunk_overlap
#             )

# # cache
#             # Initialize chunk cache
#             self.chunk_cache = ChunkCache(cache_dir=f"{cache_dir}/chunks")
# # 

#             logger.info("Loading embedding model...")

#             # self.embedding_model = EmbeddingModel(embedding_model, device=self.device) # local embeddings
# # 
#             # Initialize embedding model with caching
#             # self.embedding_model = CachedEmbeddingModel(
#             #     model_name=embedding_model,
#             #     device=device,
#             #     cache_dir=cache_dir,
#             #     # memory_cache_size=10000,
#             #     # use_disk_cache=True
#             # )

#             OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

#             self.embedding_model = GPTEmbeddingModel(
#                 model_name=embedding_model,
#                 device=device,
#                 # api_key=os.environ.get('OPENAI_API_KEY'),
#                 api_key=OPENAI_API_KEY

#                 # cache_dir=cache_dir,
#                 # memory_cache_size=10000,
#                 # use_disk_cache=True
#             )
# # 


#             logger.info("Initializing vector store...")
# # milvus
#             # self.vector_store = MilvusVectorStore(
#             #     collection_name=collection_name,
#             #     host=host,
#             #     port=port,
#             #     embedding_dim=self.embedding_model.dimension
#             # )
# # 

# # chromadb
#             self.vector_store = ChromaDBVectorStore(
#                 collection_name=collection_name,
#                 embedding_dim=self.embedding_model.dimension,
#                 persist_directory=f"{cache_dir}/vector_store"
#             )
# # 
#             OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']


#             self.hyde = HyDEGenerator()

#             # self.question_generator = QuestionGenerator()
#             # self.question_generator = GPTQuestionGenerator(api_key=os.environ.get('OPENAI_API_KEY'))
#             self.question_generator = GPTQuestionGenerator(api_key=OPENAI_API_KEY)

#             self.reranker = ReRanker()
#             self.repacker = Repacker()  # New component

#             # self.summarizer = DocumentSummarizer()
#             # self.summarizer = GPTDocumentSummarizer(api_key=os.environ.get('OPENAI_API_KEY'))
#             self.summarizer = GPTDocumentSummarizer(api_key=OPENAI_API_KEY)

#             # self.llm_processor = LLMProcessor()  # New component

#         except Exception as e:
#             logger.error(f"Pipeline initialization failed: {str(e)}")
#             raise

#     def __init__(
#             self,
#             collection_name: str = "azerbaijan_docs",
#             chunk_size: int = 512,
#             chunk_overlap: int = 50,
#             # embedding_model: str = "text-embedding-3-small",
#             embedding_model: str = "BAAI/bge-m3",
#             host: str = "localhost",
#             port: int = 19530,
#             language: str = "az",
#             device: str = None,
#             batch_size: int = 4,
#             cache_dir: str = "cache"
#         ):
#             try:
#                 self.logger = logging.getLogger(__name__)
#                 self.logger.info("Initializing RAGPipeline")
#                 st.write("Starting RAGPipeline initialization...")

#                 log_memory_usage("Starting RAGPipeline init")
    
#                 self.batch_size = batch_size
#                 self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
#                 self.language = language
#                 self.cache_dir = Path(cache_dir)
                
#                 # Create cache directory
#                 self.cache_dir.mkdir(parents=True, exist_ok=True)
#                 log_memory_usage("After cache directory setup")
#                 self.logger.info(f"Cache directory created at {self.cache_dir}")
#                 st.write(f"Cache directory setup completed")

#                 # Initialize components with status updates
#                 st.write("Initializing document processor...")
#                 self.doc_processor = DocumentProcessor()
#                 log_memory_usage("After document processor init")
#                 self.logger.info("Document processor initialized")

#                 st.write("Initializing chunking strategy...")
#                 log_memory_usage("After chunking strategy init")
#                 self.chunking_strategy = ChunkingStrategy(
#                     chunk_size=chunk_size,
#                     chunk_overlap=chunk_overlap
#                 )
#                 log_memory_usage("After chunk cache init")
#                 self.logger.info("Chunking strategy initialized")

#                 st.write("Initializing chunk cache...")
#                 self.chunk_cache = ChunkCache(cache_dir=f"{cache_dir}/chunks")
#                 self.logger.info("Chunk cache initialized")

#                 # Initialize embedding model
#                 st.write("Initializing embedding model...")
#                 self.logger.info("Loading embedding model...")
#                 OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']


# # inference api hf
#                 HUGGINGFACE_API_KEY = st.secrets['HUGGINGFACE_API_KEY']

#                 self.embedding_model = HuggingFaceEmbeddingModel(
#                     model_name="BAAI/bge-m3",
#                     api_key=st.secrets['HUGGINGFACE_API_KEY']
#                 )
# # inference api hf

#                 # Initialize embedding model with caching
#                 # self.embedding_model = CachedEmbeddingModel(
#                 #     model_name=embedding_model,
#                 #     device=device,
#                 #     cache_dir=cache_dir,
#                 #     # memory_cache_size=10000,
#                 #     # use_disk_cache=True
#                 # )

#                 # self.embedding_model = GPTEmbeddingModel(
#                 #     model_name=embedding_model,
#                 #     device=device,
#                 #     api_key=OPENAI_API_KEY
#                 # )
#                 log_memory_usage("After embedding model init")
#                 self.logger.info("Embedding model initialized")
#                 st.write("Embedding model initialized successfully")

#                 # Initialize vector store
#                 st.write("Initializing vector store...")
#                 self.logger.info("Setting up vector store...")
#                 self.vector_store = ChromaDBVectorStore(
#                     collection_name=collection_name,
#                     embedding_dim=self.embedding_model.dimension,
#                     persist_directory=f"{cache_dir}/vector_store"
#                 )
#                 log_memory_usage("After vector store init")
#                 self.logger.info("Vector store initialized")
#                 st.write("Vector store initialized successfully")

#                 # Initialize other components
#                 st.write("Initializing remaining components...")
#                 # self.hyde = HyDEGenerator()
#                 st.write("Initialized Hyde components...")
#                 self.question_generator = GPTQuestionGenerator(api_key=OPENAI_API_KEY)
#                 st.write("Initialized GPTQuestionGenerator components...")
#                 self.reranker = ReRanker()
#                 st.write("Initialized ReRanker components...")
#                 self.repacker = Repacker()
#                 st.write("Initialized Repacker components...")
#                 self.summarizer = GPTDocumentSummarizer(api_key=OPENAI_API_KEY)
#                 st.write("Initialized Summarizer components...")
                
#                 self.logger.info("All components initialized successfully")
#                 st.write("All pipeline components initialized successfully")
#                 log_memory_usage("After all components initialization")

#             except Exception as e:
#                 error_msg = f"Pipeline initialization failed: {str(e)}"
#                 self.logger.error(error_msg)
#                 st.error(error_msg)
#                 raise

    def __init__(
            self,
            collection_name: str = "azerbaijan_docs",
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            # embedding_model: str = "text-embedding-3-small",
            embedding_model: str = "BAAI/bge-m3",
            host: str = "localhost",
            port: int = 19530,
            language: str = "az",
            device: str = None,
            batch_size: int = 4,
            cache_dir: str = "cache",
            api_key: str = None  # Add this parameter
        ):
            try:
                self.logger = logging.getLogger(__name__)
                self.logger.info("Initializing RAGPipeline")
                st.write("Starting RAGPipeline initialization...")

                log_memory_usage("Starting RAGPipeline init")
    
                self.batch_size = batch_size
                self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
                self.language = language
                self.cache_dir = Path(cache_dir)
                
                # Create cache directory
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                log_memory_usage("After cache directory setup")
                self.logger.info(f"Cache directory created at {self.cache_dir}")
                st.write(f"Cache directory setup completed")

                # Initialize components with status updates
                st.write("Initializing document processor...")
                self.doc_processor = DocumentProcessor()
                log_memory_usage("After document processor init")
                self.logger.info("Document processor initialized")

                st.write("Initializing chunking strategy...")
                log_memory_usage("After chunking strategy init")
                self.chunking_strategy = ChunkingStrategy(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                log_memory_usage("After chunk cache init")
                self.logger.info("Chunking strategy initialized")

                st.write("Initializing chunk cache...")
                self.chunk_cache = ChunkCache(cache_dir=f"{cache_dir}/chunks")
                self.logger.info("Chunk cache initialized")

                # Initialize embedding model
                st.write("Initializing embedding model...")
                self.logger.info("Loading embedding model...")

                OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']


# inference api hf
                HUGGINGFACE_API_KEY = st.secrets['HUGGINGFACE_API_KEY']

                self.embedding_model = HuggingFaceEmbeddingModel(
                    model_name="BAAI/bge-m3",
                    api_key=st.secrets['HUGGINGFACE_API_KEY'],
                    device=device,
                    batch_size=batch_size,
                )
# inference api hf

                # Initialize embedding model with caching
                # self.embedding_model = CachedEmbeddingModel(
                #     model_name=embedding_model,
                #     device=device,
                #     cache_dir=cache_dir,
                #     # memory_cache_size=10000,
                #     # use_disk_cache=True
                # )

                # self.embedding_model = GPTEmbeddingModel(
                #     model_name=embedding_model,
                #     device=device,
                #     api_key=OPENAI_API_KEY
                # )
                log_memory_usage("After embedding model init")
                self.logger.info("Embedding model initialized")
                st.write("Embedding model initialized successfully")

                # Initialize vector store
                st.write("Initializing vector store...")
                self.logger.info("Setting up vector store...")
                self.vector_store = ChromaDBVectorStore(
                    collection_name=collection_name,
                    embedding_dim=self.embedding_model.dimension,
                    persist_directory=f"{cache_dir}/vector_store"
                )
                log_memory_usage("After vector store init")
                self.logger.info("Vector store initialized")
                st.write("Vector store initialized successfully")

                # Initialize other components
                st.write("Initializing remaining components...")
                # self.hyde = HyDEGenerator()
                st.write("Initialized Hyde components...")
                self.question_generator = GPTQuestionGenerator(api_key=OPENAI_API_KEY)
                st.write("Initialized GPTQuestionGenerator components...")
                self.reranker = ReRanker()
                st.write("Initialized ReRanker components...")
                self.repacker = Repacker()
                st.write("Initialized Repacker components...")
                self.summarizer = GPTDocumentSummarizer(api_key=OPENAI_API_KEY)
                st.write("Initialized Summarizer components...")
                
                self.logger.info("All components initialized successfully")
                st.write("All pipeline components initialized successfully")
                log_memory_usage("After all components initialization")

            except Exception as e:
                error_msg = f"Pipeline initialization failed: {str(e)}"
                self.logger.error(error_msg)
                st.error(error_msg)
                raise
        
    def process_document(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """Process a document through the entire pipeline"""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # # 1. Read document
            # text, metadata = self.doc_processor.read_document(file_path)
            # print(f"Document stats: {metadata}")

            # # 2. Create chunks
            # logger.info("Creating chunks")
            # chunks = self.chunking_strategy.create_chunks(text, metadata)
            
# cache
            # Check chunk cache first
            cached_chunks = self.chunk_cache.get_chunks(file_path)

            if cached_chunks:
                logger.info("Using cached document chunks")
                chunk_texts = [chunk['text'] for chunk in cached_chunks]
                chunk_metadata = [chunk['metadata'] for chunk in cached_chunks]
                num_chunks = len(cached_chunks)
            else:
                # Process document if not cached
                logger.info("Processing document and creating chunks")
                text, metadata = self.doc_processor.read_document(file_path)
                chunks = self.chunking_strategy.create_chunks(text, metadata)
                
                # Save to cache
                self.chunk_cache.save_chunks(chunks, file_path)
                
                chunk_texts = [chunk['text'] for chunk in chunks]
                # chunk_metadata = [chunk['metadata'].__dict__ for chunk in chunks]

                chunk_metadata = []
                for chunk in chunks:
                    if isinstance(chunk['metadata'], dict):
                        # If already a dict, use as is
                        chunk_metadata.append(chunk['metadata'])
                    else:
                        # Convert ChunkMetadata object to dict
                        meta_dict = {
                            'source': chunk['metadata'].source,
                            'chunk_index': chunk['metadata'].chunk_index,
                            'total_chunks': chunk['metadata'].total_chunks,
                            'start_char': chunk['metadata'].start_char,
                            'end_char': chunk['metadata'].end_char,
                            'word_count': chunk['metadata'].word_count,
                            'page_number': chunk['metadata'].page_number,
                            'section_title': chunk['metadata'].section_title,
                            'semantic_density': chunk['metadata'].semantic_density
                        }
                        chunk_metadata.append(meta_dict)
                        
                num_chunks = len(chunks)


            # Generate embeddings (will use embedding cache internally)
            embeddings = self.embedding_model.generate_embeddings(chunk_texts)

            # Add logging for debugging
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            self.logger.info(f"Number of texts: {len(chunk_texts)}")
            self.logger.info(f"Number of metadata items: {len(chunk_metadata)}")

            # Store in vector database
            self.vector_store.insert(chunk_texts, embeddings, chunk_metadata)
# 


            # # 3. Generate embeddings
            # logger.info("Generating embeddings")
            # chunk_texts = [chunk['text'] for chunk in chunks]
            # embeddings = self.embedding_model.generate_embeddings(chunk_texts)
            
            # # 4. Store in vector database
            # logger.info("Storing in vector database")
            # chunk_metadata = [chunk['metadata'].__dict__ for chunk in chunks]
            # self.vector_store.insert(chunk_texts, embeddings, chunk_metadata)
            



            # 5. Generate questions
            # logger.info("Generating questions")
            # questions = self.question_generator.generate_questions(
            #     text,
            #     language=self.language
            # )
            
            # 6. Generate summaries
            # logger.info("Generating summaries")
            # extractive_summary = self.summarizer.extractive_summarize(text)
            # abstractive_summary = self.summarizer.abstractive_summarize(text)
            
            return {
                'status': 'success',
                # 'chunks': len(chunks),
                'chunks': num_chunks, # cached version
                # # 'questions': questions,
                # # 'extractive_summary': extractive_summary,
                # 'extractive_summary': text,
                # # 'abstractive_summary': abstractive_summary,
                # 'abstractive_summary': text,

                # 'extractive_summary': chunk_texts[0] if chunk_texts else "",  # Use first chunk as summary # don't using now
                # 'abstractive_summary': chunk_texts[0] if chunk_texts else "",  # Use first chunk as summary # don't using now

                # 'metadata': metadata,
                'metadata': metadata if 'metadata' in locals() else {'source': file_path} # cached version
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def query(
        self,
        query: str,
        top_k: int = 3,
        # use_hyde: bool = True,
        use_hyde: bool = False,     # for now
        expand_queries: bool = True
    ) -> Dict[str, Any]:
        """Process a query through the pipeline"""


        # V1 without Question Variations:
        # try:

        #     # Collect all results
        #     results = []

        #     # 2. Generate embedding for original query
        #     logger.info("Generating query embeddings")
        #     query_embedding = self.embedding_model.generate_embeddings([query])[0]
            
        #     # 3. Perform hybrid search
        #     logger.info("Performing hybrid search")
        #     query_results = self.vector_store.hybrid_search(
        #         query_embedding,
        #         query,
        #         limit=top_k
        #     )
        #     results.extend(query_results)


        #     # 1. Generate hypothetical document using HyDE if enabled
        #     if use_hyde:
        #         logger.info("Generating hypothetical document")
        #         hyde_doc = self.hyde.generate_hypothesis(query)
        #         hyde_embedding = self.embedding_model.generate_embeddings([hyde_doc])[0]
            
            
        #     if use_hyde:
        #         hyde_results = self.vector_store.hybrid_search(
        #             hyde_embedding,
        #             hyde_doc,
        #             limit=top_k
        #         )
        #         results.extend(hyde_results)
    
    
        #     # Combine and deduplicate results
        #     unique_results = {result['id']: result for result in results}.values()
        #     results_list = list(unique_results)


        #     # 4. Rerank results
        #     if results_list:
        #         logger.info("Reranking results")
        #         reranked_results = self.reranker.rerank(query, results_list)
        #         print("\nReranked Results: ", reranked_results)
            
        #     # 5. Format top results
        #     formatted_results = []
        #     for idx, result in enumerate(reranked_results[:top_k], 1):
        #         formatted_results.append({
        #             'rank': idx,
        #             'text': result.get('text', ''),
        #             'score': float(result.get('rerank_score', 0.0)),  # Ensure score is a Python float
        #             'metadata': result.get('metadata', {})
        #         })
              
        #         return {
        #             'status': 'success',
        #             'results': formatted_results
        #         }

        #     else:
        #         return {
        #             'status': 'success',
        #             'results': []
        #         }
    
        # except Exception as e:
        #     logger.error(f"Error processing query: {str(e)}")
        #     return {
        #         'status': 'error',
        #         'message': str(e)
        #     }



        # V2 with Question Variations:
        try:

            # Generate query variations if enabled
            if expand_queries:
                logger.info("Generating query variations")
                all_queries = self.question_generator.generate_questions(
                    query,
                    num_questions=1,
                    language=self.language
                )
                logger.info(f"Generated variations: {all_queries}")
            else:
                all_queries = [query]

            aze_query = all_queries[0] if all_queries != '' else 'Bağışlayın, sual insan resursları mövzusu ilə əlaqəli deyil.'
            # aze_query = all_queries[0] if all_queries[0] else 'Bağışlayın, məlumat tapılmadı.'

            all_results = []
            
            # Process each query variation
            for query_variant in all_queries:
                query_embedding = self.embedding_model.generate_embeddings([query_variant])[0]
                variant_results = self.vector_store.hybrid_search(
                    query_embedding,
                    query_variant,
                    limit=top_k
                )
                
                # Add matching query info
                for result in variant_results:
                    result['matching_query'] = query_variant
                all_results.extend(variant_results)

                # Process HyDE results if enabled
                """
                if use_hyde:
                    hyde_doc = self.hyde.generate_hypothesis(query_variant)
                    hyde_embedding = self.embedding_model.generate_embeddings([hyde_doc])[0]
                    hyde_results = self.vector_store.hybrid_search(
                        hyde_embedding,
                        hyde_doc,
                        limit=top_k
                    )
                    for result in hyde_results:
                        result['matching_query'] = f"HyDE: {query_variant}"
                    all_results.extend(hyde_results)
                """

            # Deduplicate results
            unique_results = {result['id']: result for result in all_results}.values()
            results_list = list(unique_results)


            if not results_list:
                return {
                    'status': 'success',
                    'query_variations': all_queries,
                    'results': [],
                    'response': "No relevant information found."
                }


            # Rerank against original query
            if results_list:
                self.logger.info("Reranking results")
                # reranked_results = self.reranker.rerank(query, results_list)
                reranked_results = self.reranker.rerank(aze_query, results_list)
            # else:
            #     return {'status': 'success', 'query_variations': all_queries, 'results': []}

# 
            # Repack results
            self.logger.info("Repacking results")
            # repacked_data = self.repacker.repack_results(reranked_results, query)
            repacked_data = self.repacker.repack_results(reranked_results, aze_query)
            print('\nRepacked_data:', repacked_data)

            # Generate summaries
            self.logger.info("Generating summaries")
            summaries = self.summarizer.summarize_results(
                reranked_results,
                repacked_data['context'],
                # query,
                aze_query,
                language=self.language
            )

            # Generate LLM response
            # self.logger.info("Generating LLM response")
            # llm_response = self.llm_processor.generate_response(
            #     # query,
            #     aze_query,
            #     repacked_data['context'],
            #     summaries,
            #     language=self.language
            # )

# 

            # Format results
            formatted_results = []
            for idx, result in enumerate(reranked_results[:top_k], 1):
                formatted_results.append({
                    'rank': idx,
                    'text': result.get('text', ''),
                    'score': float(result.get('rerank_score', 0.0)),
                    'metadata': result.get('metadata', {}),
                    # 'matching_query': result.get('matching_query', query)
                    'matching_query': result.get('matching_query', aze_query)
                })
            
            
            # Additional:
            # Generate both types of summaries
            # summaries = self.summarizer.summarize_results(
            #     formatted_results,
            #     query,
            #     language=self.language
            # )
            
            
            return {
                'status': 'success',
                'query_variations': all_queries,
                'results': formatted_results,

                'summaries': summaries,
                # 'response': llm_response.get('response', ''),
                'context': repacked_data['context']
                
                # 'extractive_summary': summaries['extractive_summary'], # additional
                # 'abstractive_summary': summaries['abstractive_summary'] # additional
            }


        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }



    def _init_embedding_model(self, model_name):
        """Initialize embedding model with resource tracking"""
        try:
            return EmbeddingModel(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def _init_vector_store(self, **kwargs):
        """Initialize vector store with resource tracking"""
        try:
            return MilvusVectorStore(**kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    """
    def _init_hyde(self):
        # Initialize HyDE with resource tracking
        try:
            return HyDEGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize HyDE: {str(e)}")
            raise
    """    
    
    def _init_reranker(self):
        """Initialize reranker with resource tracking"""
        try:
            return ReRanker()
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {str(e)}")
            raise
    
    def _init_summarizer(self):
        """Initialize summarizer with resource tracking"""
        try:
            return DocumentSummarizer()
        except Exception as e:
            logger.error(f"Failed to initialize summarizer: {str(e)}")
            raise
    
    def _init_question_generator(self):
        """Initialize question generator with resource tracking"""
        try:
            return QuestionGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize question generator: {str(e)}")
            raise

    def __del__(self):
        """Cleanup when pipeline is destroyed"""
        ResourceManager.cleanup()








# save the question and augmented questions in excel file also // maybe main:?
# import pandas as pd
# from tqdm import tqdm
# import time
# import psutil

# # Write to excel main:
# def main():
#     """Modified main function to process questions and save question variations to Excel"""
    
#     print(f"Initial memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

#     try:
#         # Initialize pipeline
#         pipeline = RAGPipeline(
#             collection_name="azerbaijan_docs",
#             chunk_size=1024,
#             chunk_overlap=128,
#             language="az",
#             cache_dir="embeddings_cache"
#         )

#         # Process document first
#         doc_result = pipeline.process_document("formatted_combined.pdf")
        
#         if doc_result['status'] == 'success':
#             print("\nDocument Processing Results:")
#             print(f"Successfully processed document with {doc_result['chunks']} chunks")
            
#             # Read questions from Excel file
#             try:
#                 df = pd.read_excel('/Users/rahimovamir/Downloads/rag_project/HR_QA_results.xlsx')

#                 # df = df[:76]
#                 # df = df[48:52]

#                 print(f"\nLoaded {len(df)} questions from Excel file")
#             except Exception as e:
#                 print(f"Error reading Excel file: {str(e)}")
#                 return

#             # Add new columns for answers and variations if they don't exist
#             new_columns = {
#                 'original_question': '',
#                 'fixed_question': '',
#                 'augmented_questions': '',
#                 # 'extractive_answer': '',
#                 'abstractive_answer': '',
#                 'rel_context': '',
#                 # 'llm_response': ''
#             }
            
#             for col, default_value in new_columns.items():
#                 if col not in df.columns:
#                     df[col] = default_value
                
#             # Process each question
#             for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
#                 query = row['question']  # Original question
                
#                 if pd.isna(query):  # Skip if question is empty
#                     continue
                    
#                 print(f"\nProcessing question {idx + 1}: {query}")
                
#                 try:
#                     query_result = pipeline.query(
#                         query,
#                         use_hyde=False,
#                         expand_queries=True,
#                         top_k=3
#                     )
                    
#                     if query_result['status'] == 'success':
#                         # Store original question
#                         df.at[idx, 'original_question'] = query
                        
#                         # Store fixed and augmented questions
#                         if query_result['query_variations']:
#                             df.at[idx, 'fixed_question'] = query_result['query_variations'][0]
#                             if len(query_result['query_variations']) > 1:
#                                 # Join additional variations with semicolon
#                                 augmented = '; '.join(query_result['query_variations'][1:])
#                                 df.at[idx, 'augmented_questions'] = augmented
                        
#                         # Store answers
#                         # df.at[idx, 'extractive_answer'] = query_result['summaries']['extractive_summary']
#                         df.at[idx, 'abstractive_answer'] = query_result['summaries']['abstractive_summary']
#                         df.at[idx, 'rel_context'] = query_result['summaries']['relevant_info']
#                         # df.at[idx, 'llm_response'] = query_result['response']
                        
#                         # Save after each question (in case of crashes)
#                         df.to_excel('hr_questions_with_answers_and_variations_emb_tema_with_summr_version5_cached_minimized_time_chroma.xlsx', index=False)
                        
#                         # Print progress
#                         print(f"\nFixed Question: {df.at[idx, 'fixed_question']}")
#                         print(f"Augmented Questions: {df.at[idx, 'augmented_questions']}")
#                         # print(f"Extracted answer: {query_result['summaries']['extractive_summary']}")
#                         print(f"Generated answer: {query_result['summaries']['abstractive_summary']}")
#                         print(f"\nRelevant Context: {query_result['summaries']['relevant_info']}")
#                         # print(f"LLM Response: {query_result['response']}")
                        
#                     else:
#                         print(f"Error processing query: {query_result.get('message', 'Unknown error')}")
                        
#                     # Add a small delay between questions
#                     time.sleep(1)
                    
#                 except Exception as e:
#                     print(f"Error processing question {idx + 1}: {str(e)}")
#                     continue
                
#                 # Print memory usage periodically
#                 if idx % 10 == 0:
#                     print(f"\nCurrent memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            
#             # Final save
#             print("\nSaving final results to Excel...")
#             df.to_excel('hr_questions_with_answers_and_variations_emb_tema_with_summr_version5_cached_minimized_time_chroma.xlsx', index=False)
#             print("Results saved successfully!")
            
#         else:
#             print(f"Error in document processing: {doc_result['message']}")
     
#     except Exception as e:
#         logger.error(f"Pipeline execution failed: {str(e)}")
#     finally:
#         # Ensure cleanup
#         ResourceManager.cleanup()











# With using streamlit (without cache):
# import streamlit as st
# import torch
# import logging
# from typing import List, Dict, Any
# import time

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def initialize_rag_pipeline():
#     """Initialize RAG pipeline with proper error handling"""
#     try:
#         pipeline = RAGPipeline(
#             collection_name="azerbaijan_docs",
#             chunk_size=1024,
#             chunk_overlap=128,
#             language="az",
#             cache_dir="embeddings_cache"
#         )
        
#         # Process initial document
#         doc_result = pipeline.process_document("formatted_combined.pdf")
#         if doc_result['status'] != 'success':
#             st.error(f"Error processing document: {doc_result.get('message', 'Unknown error')}")
#             return None
            
#         return pipeline
        
#     except Exception as e:
#         st.error(f"Failed to initialize pipeline: {str(e)}")
#         return None

# def process_query(pipeline, query: str) -> Dict[str, Any]:
#     """Process a single query through the pipeline"""
#     try:
#         return pipeline.query(
#             query,
#             use_hyde=False,
#             expand_queries=True,
#             top_k=3
#         )
#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         return {
#             'status': 'error',
#             'message': str(e)
#         }

# def main():
#     st.set_page_config(
#         page_title="HR Assistant",
#         page_icon="👩‍💼",
#         layout="wide"
#     )

#     st.title("HR Policy Assistant")
#     st.markdown("""
#     Welcome to the HR Policy Assistant! Ask any questions about HR policies and procedures.
#     """)

#     # Initialize session state for chat history
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     # Initialize RAG pipeline
#     if 'pipeline' not in st.session_state:
#         with st.spinner("Initializing AI model..."):
#             pipeline = initialize_rag_pipeline()
#             if pipeline:
#                 st.session_state.pipeline = pipeline
#                 st.success("AI model initialized successfully!")
#             else:
#                 st.error("Failed to initialize AI model. Please refresh the page.")
#                 return

#     # Create two columns
#     col1, col2 = st.columns([2, 1])

#     with col1:
#         # Chat interface
#         st.subheader("Chat")
        
#         # Display chat history
#         for message in st.session_state.chat_history:
#             if message["role"] == "user":
#                 st.markdown(f"**You:** {message['content']}")
#             else:
#                 st.markdown(f"**Assistant:** {message['content']}")
#                 if "context" in message:
#                     with st.expander("View Context"):
#                         st.markdown(message["context"])

#         # Query input
#         query = st.text_input("Ask a question:", key="query_input")
        
#         if st.button("Send", key="send_button"):
#             if not query:
#                 st.warning("Please enter a question.")
#                 return
                
#             # Add user message to chat history
#             st.session_state.chat_history.append({
#                 "role": "user",
#                 "content": query
#             })

#             # Process query
#             with st.spinner("Processing your question..."):
#                 result = process_query(st.session_state.pipeline, query)

#             if result['status'] == 'success':
#                 # Get the main response
#                 response = result.get('response', '')
#                 if not response:  # Fallback to abstractive summary if no LLM response
#                     response = result.get('summaries', {}).get('abstractive_summary', '')

#                 # Add assistant response to chat history
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": response,
#                     "context": result.get('summaries', {}).get('relevant_info', '')
#                 })

#                 # Force streamlit to rerun to update chat history
#                 st.rerun()
#             else:
#                 st.error(f"Error: {result.get('message', 'Unknown error')}")

#     with col2:
#         # Information panel
#         st.subheader("Information")
#         st.markdown("""
#         ### Tips for better results:
#         - Be specific in your questions
#         - Use clear and concise language
#         - Ask one question at a time
#         - Questions can be in Azerbaijani or English
#         """)

#         # Clear chat button
#         if st.button("Clear Chat History"):
#             st.session_state.chat_history = []
#             st.rerun()







# # MAIN streamlit (with cached version):

import streamlit as st
import torch
import logging
from typing import List, Dict, Any
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def log_memory_usage(message: str = ""):
    """Log current memory usage with optional context message"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    # Log to both streamlit and logger
    log_message = f"Memory usage{f' ({message})' if message else ''}: {memory_mb:.2f} MB"
    logger.info(log_message)
    st.write(log_message)



# def initialize_rag_pipeline():
#     """Initialize RAG pipeline with proper error handling and caching"""
#     try:
#         # Create cache directories if they don't exist
#         cache_dir = Path("cache")
#         cache_dir.mkdir(parents=True, exist_ok=True)
#         for subdir in ["model", "chunks", "embeddings"]:
#             (cache_dir / subdir).mkdir(exist_ok=True)

#         st.write("Setting up components...")

#         # Initialize pipeline with caching enabled
#         pipeline = RAGPipeline(
#             collection_name="azerbaijan_docs",
#             embedding_model="text-embedding-3-small",  # OpenAI model
#             chunk_size=1024,
#             chunk_overlap=128,
#             language="az",
#             cache_dir=str(cache_dir),
#             batch_size=4  # Smaller batch size for smoother operation
#         )
        
#         st.write("Rag pipeline is setuped")

#         # Process initial document (will use cache if available)
#         doc_result = pipeline.process_document("formatted_combined.pdf")
#         if doc_result['status'] != 'success':
#             st.error(f"Error processing document: {doc_result.get('message', 'Unknown error')}")
#             return None
            
#         return pipeline
        
#     except Exception as e:
#         st.error(f"Failed to initialize pipeline: {str(e)}")
#         return None

def initialize_rag_pipeline():
    """Initialize RAG pipeline with detailed logging and status updates"""
    try:
        # Create cache directories with status updates
        cache_dir = Path("cache")
        st.write("Creating cache directories...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["model", "chunks", "embeddings", "vector_store"]:
            (cache_dir / subdir).mkdir(exist_ok=True)
            st.write(f"Created cache directory: {subdir}")

        st.write("Initializing pipeline components...")
        logger.info("Starting pipeline initialization")
        log_memory_usage("Before initialization")

        HUGGINGFACE_API_KEY = st.secrets['HUGGINGFACE_API_KEY']

        # Initialize components with detailed status updates
        pipeline = RAGPipeline(
            collection_name="azerbaijan_docs",
            # embedding_model="text-embedding-3-small",
            embedding_model="BAAI/bge-m3", # Local model
            chunk_size=1024,
            chunk_overlap=128,
            language="az",
            cache_dir=str(cache_dir),
            batch_size=4,
            api_key=HUGGINGFACE_API_KEY  # Add this line
        )
        
        log_memory_usage("After creating cache directories")
        st.write("Pipeline object created successfully")
        logger.info("Pipeline object initialized")

        # Process initial document with progress updates
        log_memory_usage("After pipeline initialization")
        st.write("Processing initial document...")
        logger.info("Starting document processing")
        
        try:
            doc_result = pipeline.process_document("formatted_combined.pdf")
            log_memory_usage("After document processing")
            st.write("Document processing completed")
            logger.info("Document processing finished")
            
            if doc_result['status'] != 'success':
                error_msg = f"Error processing document: {doc_result.get('message', 'Unknown error')}"
                st.error(error_msg)
                logger.error(error_msg)
                return None
                
            st.success("Pipeline initialization completed successfully!")
            logger.info("Pipeline fully initialized")
            log_memory_usage("After complete initialization")
            return pipeline
            
        except Exception as doc_error:
            error_msg = f"Document processing failed: {str(doc_error)}"
            st.error(error_msg)
            logger.error(error_msg)
            return None
        
    except Exception as e:
        error_msg = f"Failed to initialize pipeline: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)
        return None



def process_query(pipeline, query: str) -> Dict[str, Any]:
    """Process a single query through the pipeline"""
    try:
        result = pipeline.query(
            query,
            use_hyde=False,
            expand_queries=True,
            top_k=3
        )
        
        # Format the response
        if result['status'] == 'success':
            # Get the most relevant answer
            response = result.get('summaries', {}).get('abstractive_summary', '')
            context = result.get('summaries', {}).get('relevant_info', '')
            
            # Get query variations if available
            variations = result.get('query_variations', [])
            
            return {
                'status': 'success',
                'response': response,
                'context': context,
                'variations': variations
            }
        else:
            return result
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

def format_chat_message(message: dict) -> None:
    """Format and display a chat message"""
    if message["role"] == "user":
        st.markdown(f"🧑‍💼 **You:** {message['content']}")
    else:
        st.markdown(f"🤖 **Assistant:** {message['content']}")
        if message.get("variations"):
            with st.expander("Question Variations"):
                for var in message["variations"]:
                    st.markdown(f"- {var}")
        if message.get("context"):
            with st.expander("View Context"):
                st.markdown(message["context"])

def main():
    # Set up Streamlit page configuration
    st.set_page_config(
        page_title="HR Assistant",
        page_icon="👩‍💼",
        layout="wide"
    )

    # Apply custom CSS for better styling
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #f0f2f6;
        }
        .assistant-message {
            background-color: #e8f0fe;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("HR Policy Assistant")
    st.markdown("""
    Welcome to the HR Policy Assistant! Ask any questions about HR policies and procedures.
    This system uses advanced AI to provide accurate answers based on official HR documentation.
    """)

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Initialize RAG pipeline if not already initialized
        if st.session_state.pipeline is None:
            with st.spinner("Initializing AI model (this may take a few moments)..."):
                pipeline = initialize_rag_pipeline()
                if pipeline:
                    st.session_state.pipeline = pipeline
                    st.success("AI model initialized successfully!")
                else:
                    st.error("Failed to initialize AI model. Please refresh the page.")
                    return

        # Chat interface
        st.subheader("Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            format_chat_message(message)

        # Query input
        with st.form(key="query_form"):
            query = st.text_input(
                "Ask a question:",
                key="query_input",
                placeholder="Type your question here..."
            )
            submit_button = st.form_submit_button("Send")

            if submit_button and query:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": query
                })

                # Process query
                with st.spinner("Processing your question..."):
                    result = process_query(st.session_state.pipeline, query)

                if result['status'] == 'success':
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['response'],
                        "context": result.get('context', ''),
                        "variations": result.get('variations', [])
                    })

                    # Force streamlit to rerun to update chat history
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('message', 'Unknown error')}")

    with col2:
        # Information panel
        st.subheader("Information")
        st.markdown("""
        ### Tips for better results:
        - Be specific in your questions
        - Use clear and concise language
        - Ask one question at a time
        - Questions can be in Azerbaijani or English
        
        ### About the System:
        This assistant uses advanced natural language processing to:
        - Understand and rephrase questions
        - Search through HR documentation
        - Generate accurate, context-aware responses
        """)

        # System status
        st.subheader("System Status")
        status_placeholder = st.empty()
        status_placeholder.success("System is ready")

        # Clear chat button with confirmation
        if st.button("Clear Chat History"):
            if st.session_state.chat_history:
                st.session_state.chat_history = []
                st.rerun()






if __name__ == "__main__":
    main()




