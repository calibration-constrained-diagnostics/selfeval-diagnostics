import logging
import torch
import boto3
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model class definitions
BEDROCK_MODEL_IDS = {
    'deepseek_r1': 'us.deepseek.r1-v1:0',
    'nova_premier': 'us.amazon.nova-premier-v1:0',
    'c37_sonnet': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
}

class LLMModel:
    """Base class for language models with common interface"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate(self, prompt):
        """Generate text from prompt"""
        raise NotImplementedError

    def format_chat_prompt(self, system_prompt, user_prompt):
        """Format system and user prompts"""
        return f"{system_prompt}\n\n{user_prompt}"
    
    def set_context(self, context):
        """Set context for the model"""
        pass

class HFModel(LLMModel):
    """Hugging Face model implementation"""
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context = ""
        
        self.logger.info(f"Loading model: {model_name}, device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
    def generate(self, prompt):
        """Generate text from prompt"""
        # Add context if available
        if self.context and not prompt.startswith(self.context):
            full_prompt = self.context + "\n\n" + prompt
        else:
            full_prompt = prompt
            
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response
        
    def format_chat_prompt(self, system_prompt, user_prompt):
        """Format model-specific chat prompts"""
        if "llama" in self.model_name.lower():
            return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        elif "mistral" in self.model_name.lower():
            return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        elif "qwen" in self.model_name.lower():
            chat_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            return f"{system_prompt}\n\n{user_prompt}\n\nAssistant:"
    
    def set_context(self, context):
        """Set context for in-context learning"""
        self.context = context
        self.logger.info(f"Context set ({len(context)} chars)")

class BedrockModel(LLMModel):
    """AWS Bedrock model implementation using converse API"""
    
    def __init__(self, model_id):
        super().__init__()
        self.model_id = BEDROCK_MODEL_IDS.get(model_id, model_id)
        self.model_name = model_id
        self.context = ""
        
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.logger.info(f"Initialized Bedrock model: {self.model_id}")
    
    def generate(self, prompt):
        """Generate text using Bedrock converse API"""
        return self._call_converse_api(prompt)
    
    def _call_converse_api(self, prompt):
        """
        Call Bedrock converse API with proper message formatting and retry logic
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        # Add system message if context is available
        if self.context:
            messages.insert(0, {"role": "system", "content": [{"text": self.context}]})
        
        # Add retry logic - attempt up to 5 times
        max_retries = 7
        retry_count = 0
        backoff_time = 2  # Start with 1 second backoff
        
        while retry_count < max_retries:
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    inferenceConfig={
                        "maxTokens": 4096,
                        "temperature": 0.7,
                    }
                )
                
                result = response['output']['message']['content'][0]['text']
                self.logger.debug(f"Generated response ({len(result)} chars)")
                return result
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.warning(f"Converse API error (attempt {retry_count}/{max_retries}): {str(e)}")
                    self.logger.info(f"Retrying in {backoff_time} seconds...")
                    # time.sleep(backoff_time)
                    # Implement exponential backoff
                    backoff_time *= 2
                else:
                    self.logger.error(f"Converse API failed after {max_retries} attempts: {str(e)}")
                    raise e
    
    def set_context(self, context):
        """Set context for system message"""
        self.context = context
        self.logger.info(f"Context set for Bedrock model ({len(context)} chars)")