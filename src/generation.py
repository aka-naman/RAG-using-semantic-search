import os
import torch
import yaml
from typing import List, Dict, Any, Optional
from openai import OpenAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from src.ingestion import DocumentChunk

class RAGGenerator:
    """Base class for RAG generation."""
    def __init__(self, prompt_file: str = "prompts.yaml"):
        self.prompt_file = prompt_file
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, Any]:
        """Loads prompts from a YAML file."""
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"Prompt file {self.prompt_file} not found.")
        with open(self.prompt_file, "r") as f:
            return yaml.safe_load(f).get("rag_system", {})

    def generate_answer(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        raise NotImplementedError
    
    def self_correct(self, query: str, context_chunks: List[DocumentChunk], answer: str) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIRAGGenerator(RAGGenerator):
    def __init__(self, model_name: str = "gpt-4o", prompt_file: str = "prompts.yaml"):
        super().__init__(prompt_file)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate_answer(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        context_text = "\n\n".join([f"[Source: {c.metadata.get('source', 'Unknown')}] {c.content}" for c in context_chunks])
        template = self.prompts.get("generation_prompt")
        prompt = template.format(context=context_text, query=query)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def self_correct(self, query: str, context_chunks: List[DocumentChunk], answer: str) -> Dict[str, Any]:
        context_text = "\n\n".join([f"[Source: {c.metadata.get('source', 'Unknown')}] {c.content}" for c in context_chunks])
        template = self.prompts.get("self_correction_prompt")
        prompt = template.format(context=context_text, query=query, answer=answer)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return {"evaluation": response.choices[0].message.content}

class LocalRAGGenerator(RAGGenerator):
    def __init__(self, model_id: str = "unsloth/Llama-3.2-1B-Instruct", prompt_file: str = "prompts.yaml"):
        """Uses HuggingFace Transformers for offline generation."""
        super().__init__(prompt_file)
        print(f"Loading local model: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map="auto"
        )
        self.pipe = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def generate_answer(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        context_text = "\n\n".join([f"[Source: {c.metadata.get('source', 'Unknown')}] {c.content}" for c in context_chunks])
        template = self.prompts.get("generation_prompt")
        # Ensure we have placeholders
        prompt_content = template.format(context=context_text, query=query)
        
        messages = [
            {"role": "system", "content": "You are a professional technical assistant."},
            {"role": "user", "content": prompt_content}
        ]
        
        # Format prompt using chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = self.pipe(
            prompt, 
            max_new_tokens=512, 
            temperature=0.1, 
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return outputs[0]['generated_text'][len(prompt):].strip()

    def self_correct(self, query: str, context_chunks: List[DocumentChunk], answer: str) -> Dict[str, Any]:
        """Simplified local self-correction."""
        return {"evaluation": "Self-correction is currently limited in local mode to save resources."}

if __name__ == "__main__":
    print("RAGGenerator classes defined.")
