"""
Text Corrector using LLMs
使用大语言模型修复扰动文本（零训练方法）
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import os


class TextCorrector:
    """使用LLM修复扰动文本"""
    
    def __init__(self, method='qwen', model_path=None, api_key=None):
        """
        初始化文本修复器
        
        Args:
            method: 'qwen', 'gpt', 'llama'
            model_path: 本地模型路径（qwen/llama）
            api_key: OpenAI API密钥（gpt）
        """
        self.method = method
        
        if method == 'qwen':
            self._init_qwen(model_path)
        elif method == 'gpt':
            self._init_gpt(api_key)
        elif method == 'llama':
            self._init_llama(model_path)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _init_qwen(self, model_path):
        """初始化Qwen模型"""
        if model_path is None:
            model_path = "Qwen/Qwen2.5-3B-Instruct"
        
        print(f"Loading Qwen model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("✓ Qwen model loaded successfully!")
    
    def _init_gpt(self, api_key):
        """初始化GPT API"""
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = api_key
        self.model_name = "gpt-3.5-turbo"
        print(f"✓ Initialized OpenAI API with model: {self.model_name}")
    
    def _init_llama(self, model_path):
        """初始化Llama模型"""
        if model_path is None:
            model_path = "meta-llama/Llama-2-7b-chat-hf"
        
        print(f"Loading Llama model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        print("✓ Llama model loaded successfully!")
    
    def create_correction_prompt(self, text: str) -> str:
        """
        创建修复prompt
        
        Args:
            text: 待修复的文本
        
        Returns:
            prompt: 完整的prompt
        """
        prompt = f"""You are an expert at fixing severely corrupted text captions.

Fix ALL errors including character substitutions, spelling mistakes, and typos.

Examples:
Input: "remove multiple p1ns and cahnge the background to gray"
Output: "remove multiple pins and change the background to gray"

Input: "$!ow a d0g sleep1ng wint a pupp7"
Output: "show a dog sleeping with a puppy"

Input: "the lunch room has a wite sofa, wider angle pic"
Output: "the lunch room has a white sofa, wider angle pic"

Now fix this (output ONLY the corrected text):
Input: "{text}"
Output:"""
        
        return prompt
    
    def correct_text_qwen(self, text: str, max_new_tokens=100) -> str:
        """使用Qwen修复文本"""
        prompt = self.create_correction_prompt(text)
        
        messages = [
            {"role": "system", "content": "You are a helpful text correction assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取修复后的文本（去除多余内容）
        corrected = response.strip().strip('"').strip()
        
        return corrected
    
    def correct_text_gpt(self, text: str) -> str:
        """使用GPT修复文本"""
        import openai
        
        prompt = self.create_correction_prompt(text)
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful text correction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        corrected = response['choices'][0]['message']['content'].strip().strip('"').strip()
        return corrected
    
    def correct_text_llama(self, text: str, max_new_tokens=100) -> str:
        """使用Llama修复文本"""
        prompt = self.create_correction_prompt(text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的部分
        corrected = response[len(prompt):].strip().strip('"').strip()
        
        return corrected
    
    def correct_single(self, text: str) -> Tuple[str, bool]:
        """
        修复单个文本
        
        Returns:
            corrected_text: 修复后的文本
            is_changed: 是否发生修改
        """
        try:
            if self.method == 'qwen':
                corrected = self.correct_text_qwen(text)
            elif self.method == 'gpt':
                corrected = self.correct_text_gpt(text)
            elif self.method == 'llama':
                corrected = self.correct_text_llama(text)
            
            is_changed = (text.strip() != corrected.strip())
            
            return corrected, is_changed
        except Exception as e:
            print(f"Error correcting text: {e}")
            return text, False
    
    def correct_batch(self, texts: List[str], verbose=False) -> List[str]:
        """
        批量修复文本
        
        Args:
            texts: 待修复的文本列表
            verbose: 是否打印修复过程
        
        Returns:
            corrected_texts: 修复后的文本列表
        """
        corrected_texts = []
        
        for i, text in enumerate(texts):
            corrected, is_changed = self.correct_single(text)
            corrected_texts.append(corrected)
            
            if verbose and is_changed:
                print(f"\n[{i+1}] Original:  {text}")
                print(f"    Corrected: {corrected}")
        
        return corrected_texts


def demo():
    """演示示例"""
    print("="*70)
    print("Text Corrector Demo")
    print("="*70)
    
    # 初始化修复器
    corrector = TextCorrector(method='qwen')
    
    # 测试样本
    test_texts = [
        "remove multiple pins and cahnge the background to gray",
        "the lunch room has a wite sofa, wider angle pic",
        "Put a clock on the wall to the left of the door."  # 正常文本
    ]
    
    for text in test_texts:
        corrected, is_changed = corrector.correct_single(text)
        
        print(f"\nOriginal:  {text}")
        print(f"Corrected: {corrected}")
        print(f"Changed:   {'Yes ✓' if is_changed else 'No'}")
        print("-"*70)


if __name__ == "__main__":
    demo()




