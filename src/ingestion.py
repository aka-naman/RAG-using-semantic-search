import os
import pandas as pd
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions, EasyOcrOptions, PipelineOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]

class IngestionPipeline:
    def __init__(self):
        # Configure EasyOCR as the engine (more stable on low memory than default)
        ocr_options = EasyOcrOptions()
        ocr_options.use_gpu = False # Keep on CPU to save VRAM for the LLM
        
        # PDF Pipeline Options
        pdf_options = PdfPipelineOptions()
        pdf_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
        pdf_options.do_ocr = True 
        pdf_options.ocr_options = ocr_options
        
        # Image Pipeline Options
        # In Docling v2.80, ImageFormatOption uses PipelineOptions which has accelerator_options but NO ocr_options field directly
        image_options = PipelineOptions()
        image_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_options)
        }
        
        self.converter = DocumentConverter(
            format_options=format_options
        )

    def load_documents(self, doc_dir: str) -> List[DocumentChunk]:
        """Loads various document types using Docling for AI-powered extraction."""
        documents = []
        
        if not os.path.exists(doc_dir):
            os.makedirs(doc_dir)
            return documents

        # Formats supported by Docling
        docling_extensions = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".asciidoc", ".md", ".png", ".jpg", ".jpeg"]

        for filename in os.listdir(doc_dir):
            file_path = os.path.join(doc_dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            try:
                if ext in docling_extensions:
                    print(f"Converting {filename} with Docling...")
                    result = self.converter.convert(file_path)
                    content = result.document.export_to_markdown()
                    documents.append(DocumentChunk(
                        content=content,
                        metadata={"source": filename, "type": ext[1:], "method": "docling"}
                    ))
                elif ext == ".txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        documents.append(DocumentChunk(
                            content=text,
                            metadata={"source": filename, "type": "txt", "method": "manual"}
                        ))
                elif ext == ".csv":
                    df = pd.read_csv(file_path)
                    content = "Table Data:\n" + df.to_markdown(index=False)
                    documents.append(DocumentChunk(
                        content=content,
                        metadata={"source": filename, "type": "csv", "method": "pandas"}
                    ))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                    
        return documents

def load_documents(doc_dir: str) -> List[DocumentChunk]:
    pipeline = IngestionPipeline()
    return pipeline.load_documents(doc_dir)

def chunk_documents(documents: List[DocumentChunk], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[DocumentChunk]:
    """Chunks documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunked_docs = []
    for doc in documents:
        if doc.metadata.get("method") == "pandas" and len(doc.content) < chunk_size * 1.5:
            chunked_docs.append(DocumentChunk(
                content=doc.content,
                metadata={**doc.metadata, "chunk_id": 0}
            ))
        else:
            chunks = text_splitter.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                chunked_docs.append(DocumentChunk(
                    content=chunk,
                    metadata={**doc.metadata, "chunk_id": i}
                ))
            
    return chunked_docs
