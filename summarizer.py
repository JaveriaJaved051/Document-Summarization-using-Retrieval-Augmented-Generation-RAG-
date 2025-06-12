import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Llama3Summarizer:
    def __init__(self, model_name: str):
        # Configure 4-bit quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    def _create_prompt(self, context: str) -> str:
        """Construct optimized prompt for LLaMA 3 summarization"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert document summarization assistant. Your task is to generate a concise, coherent, 
and accurate summary based on the provided context. Follow these guidelines:
1. Capture the core ideas and key information
2. Maintain factual accuracy
3. Use clear and concise language
4. Structure the summary logically
5. Preserve important technical terms
6. Omit trivial details and examples
7. Keep the summary under 250 words<|eot_id|>
<|start_header_id|>user<|end_header_id|>
DOCUMENT CONTEXT:
{context}

Generate a comprehensive summary following the specified guidelines.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
SUMMARY:"""
    
    def generate_summary(self, context: str) -> str:
        prompt = self._create_prompt(context)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # LLaMA 3 context window
        ).to(self.model.device)
        
        # Generate summary with optimized parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        # Decode and clean the output
        summary = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return summary.strip()