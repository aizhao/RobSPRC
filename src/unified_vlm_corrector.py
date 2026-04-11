"""
Unified VLM Corrector with Built-in PPL Detection
统一的VLM修复器（内置PPL检测）
使用同一个模型进行困惑度检测和文本修复
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from pathlib import Path


class UnifiedVLMCorrector:
    """
    统一的VLM修复器
    使用同一个模型进行PPL检测和文本修复
    支持：Qwen2-VL-2B, Qwen-VL-Chat, Qwen2.5-3B
    """
    
    def __init__(
        self, 
        model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        ppl_threshold: float = 50.0,
        max_iterations: int = 3,
        device: str = 'cuda'
    ):
        """
        初始化统一修复器
        
        Args:
            model_path: 模型路径
                - "Qwen/Qwen2-VL-2B-Instruct" (推荐，支持图像，2B参数)
                - "Qwen/Qwen-VL-Chat" (支持图像，7B参数)
                - "Qwen/Qwen2.5-3B-Instruct" (纯文本，3B参数)
            ppl_threshold: PPL阈值
            max_iterations: 最大迭代次数
            device: 设备
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.ppl_threshold = ppl_threshold
        self.max_iterations = max_iterations
        
        # 判断模型类型
        self.is_vlm = 'VL' in model_path or 'vl' in model_path
        
        print(f"\n{'='*70}")
        print(f"Initializing Unified VLM Corrector")
        print(f"Model: {model_path}")
        print(f"Type: {'Vision-Language Model' if self.is_vlm else 'Text-Only Model'}")
        print(f"PPL Threshold: {ppl_threshold}")
        print(f"Max Iterations: {max_iterations}")
        print(f"{'='*70}\n")
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if self.is_vlm:
            self._load_vlm()
        else:
            self._load_text_model()
    
    def _load_vlm(self):
        """加载视觉语言模型"""
        print(f"Loading VLM model: {self.model_path}")
        
        try:
            # Qwen2-VL 需要特殊处理
            if 'Qwen2-VL' in self.model_path or 'qwen2-vl' in self.model_path:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                
                print("Detected Qwen2-VL model, using Qwen2VLForConditionalGeneration...")
                
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                self.tokenizer = self.processor.tokenizer
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                ).eval()
                
                self.vlm_type = 'qwen2-vl'
                
            # Qwen-VL (旧版)
            elif 'Qwen-VL' in self.model_path or 'qwen-vl' in self.model_path:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                print("Detected Qwen-VL model, using AutoModelForCausalLM...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                ).eval()
                
                self.vlm_type = 'qwen-vl'
            
            else:
                raise ValueError(f"Unknown VLM type: {self.model_path}")
            
            print("✓ VLM model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading VLM: {e}")
            print("Please use a text-only model like Qwen/Qwen2.5-3B-Instruct")
            raise
    
    def _load_text_model(self):
        """加载纯文本模型"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading text model: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✓ Text model loaded successfully!")
    
    @torch.no_grad()
    def compute_perplexity(
        self, 
        texts: List[str], 
        batch_size: int = 8,
        verbose: bool = True
    ) -> np.ndarray:
        """
        计算文本列表的困惑度
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            verbose: 是否显示进度
        
        Returns:
            ppls: 困惑度数组
        """
        ppls = []
        
        if verbose:
            print(f"Computing PPL for {len(texts)} samples with {self.model_path.split('/')[-1]}...")
        
        iterator = tqdm(range(0, len(texts), batch_size), desc="Computing PPL") if verbose else range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # 过滤空文本
            valid_texts = [t if t.strip() else " " for t in batch_texts]
            
            try:
                # Tokenize
                encodings = self.tokenizer(
                    valid_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # 移动到设备
                input_ids = encodings.input_ids.to(self.model.device)
                attention_mask = encodings.attention_mask.to(self.model.device)
                
                # Forward pass - 处理不同模型类型
                if self.is_vlm and hasattr(self, 'vlm_type') and self.vlm_type == 'qwen2-vl':
                    # Qwen2-VL 使用文本模型部分计算 PPL
                    # 获取语言模型部分
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                        # 使用 embedding 和语言模型部分
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                    else:
                        # 如果结构不同，直接使用模型
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                else:
                    # 普通语言模型
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                
                # 计算每个样本的PPL
                logits = outputs.logits  # [batch, seq_len, vocab]
                
                for j in range(len(batch_texts)):
                    # 获取当前样本的attention mask
                    sample_attention_mask = attention_mask[j]
                    valid_length = sample_attention_mask.sum().item()
                    
                    if valid_length <= 1:
                        ppls.append(np.nan)
                        continue
                    
                    # 计算该样本的损失
                    shift_logits = logits[j, :-1, :].contiguous()
                    shift_labels = input_ids[j, 1:].contiguous()
                    shift_mask = sample_attention_mask[1:].contiguous()
                    
                    # 计算交叉熵
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # 只计算有效token的损失
                    loss = loss.view(-1)
                    valid_loss = (loss * shift_mask.float()).sum() / shift_mask.sum()
                    
                    # PPL = exp(loss)
                    ppl = torch.exp(valid_loss).item()
                    ppls.append(ppl)
                    
            except Exception as e:
                print(f"Error computing PPL for batch {i}: {e}")
                import traceback
                traceback.print_exc()
                ppls.extend([np.nan] * len(batch_texts))
        
        return np.array(ppls)
    
    def create_correction_prompt(self, corrupted_text: str, use_image: bool = False) -> str:
        """
        创建修复prompt（针对CIR修改指令，涵盖多种扰动类型）
        
        Args:
            corrupted_text: 被扰动的文本
            use_image: 是否使用图像引导
        
        Returns:
            prompt: 指令prompt
        """
        if use_image and self.is_vlm:
            prompt = f"""This image shows a reference object. The following text is a MODIFICATION INSTRUCTION (not an image description) that has been adversarially perturbed.

The instruction describes how to modify/change the reference image to get a target image.

Corrupted instruction: "{corrupted_text}"

The text may contain various types of perturbations:
1. Character-level: typos, substitutions ($→s, 0→o), insertions, deletions, swaps
2. Word-level: synonym/antonym replacements, word order changes, missing/extra words
3. Case changes: incorrect capitalization

Your task: Restore the original instruction while preserving its modification intent.

IMPORTANT: 
- This is a modification instruction (e.g., "change X to Y", "make it more X"), NOT an image description
- Fix ALL errors: spelling, grammar, word choice, word order, capitalization
- Keep the modification intent exactly the same
- Output ONLY the corrected instruction

Corrected instruction:"""
        else:
            prompt = f"""You are an expert AI text restoration specialist for Computer Vision datasets.
The following text is a MODIFICATION INSTRUCTION for Composed Image Retrieval (CIR). It describes how to modify a reference image to get a target image.

The instruction has been heavily corrupted with adversarial noise.

Corrupted instruction: "{corrupted_text}"

Your goal is to infer the user's original intent and output the clean, grammatically correct English sentence.

Analyze the text for these specific corruption patterns found in the dataset:
1. Leet Speak & Digits: Numbers replacing letters (e.g., '0'='o', '1'='l' or 'i', '3'='e', '7'='t').
2. Symbol Injection: Random symbols inserted inside words (e.g., "M&a)ke" -> "Make").
3. Scrambled/Typo: Phonetic errors or OCR-like mistakes (e.g., "Tragte" -> "Target").
4. Word Order: Words might be slightly out of order (e.g., "background the" -> "the background").

IMPORTANT GUIDELINES:
- Context is Key: These are image editing commands. Common verbs include: "Change", "Make", "Add", "Remove", "Show", "Replace".
- Domain: The content usually involves animals (dogs, cats), objects (bottles, cars), or scenes (beach, snow).
- Do not explain or apologize. Just provide the fixed string.

Reference Examples (Learning from Data):
- Input: "Tragte on group of buffalos floating in idtechd lake"
  Output: "Target on group of buffalos floating in ditched lake"
- Input: "M&a)ke the truaczt2or bigger and add t@ree$s."
  Output: "Make the tractor bigger and add trees."
- Input: "has bIlac2k details on the bor(dge^r iside&s"
  Output: "has black details on the border sides"
- Input: "The TauUev IiJge has two sU7vwigs of a qpicHd ieIbed on two p@yt6s."
  Output: "The Target Image has two servings of a quiche served on two plates."

Corrected:"""
        
        return prompt
    
    @torch.no_grad()
    def correct_text(
        self, 
        text: str, 
        image_path: Optional[str] = None,
        max_new_tokens: int = 100
    ) -> str:
        """
        修复文本
        
        Args:
            text: 待修复文本
            image_path: 可选的参考图像路径
            max_new_tokens: 最大生成长度
        
        Returns:
            corrected: 修复后的文本
        """
        use_image = (image_path is not None) and self.is_vlm
        prompt = self.create_correction_prompt(text, use_image)
        
        try:
            if use_image and hasattr(self, 'vlm_type'):
                # Qwen2-VL 图像引导修复
                if self.vlm_type == 'qwen2-vl':
                    from PIL import Image as PILImage
                    from qwen_vl_utils import process_vision_info
                    
                    # 构建消息
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    # 处理输入
                    text_input = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text_input],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
                    inputs = inputs.to(self.model.device)
                    
                    # 生成
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.2,
                        do_sample=True
                    )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    response = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                
                # Qwen-VL (旧版)
                elif self.vlm_type == 'qwen-vl':
                    query = self.tokenizer.from_list_format([
                        {'image': image_path},
                        {'text': prompt},
                    ])
                    
                    response, _ = self.model.chat(
                        self.tokenizer, 
                        query=query, 
                        history=None,
                        max_new_tokens=max_new_tokens,
                        temperature=0.2
                    )
                else:
                    # 未知类型，fallback 到纯文本
                    use_image = False
            
            if not use_image:
                # Pure text correction
                messages = [
                    {"role": "system", "content": "You are an expert text correction assistant."},
                    {"role": "user", "content": prompt}
                ]
                
                text_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
                
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up response
            corrected = response.strip().strip('"').strip()
            
            # 移除可能的前缀
            prefixes = ['corrected caption:', 'corrected:', 'output:']
            for prefix in prefixes:
                if corrected.lower().startswith(prefix):
                    corrected = corrected[len(prefix):].strip()
            
            return corrected if corrected else text
            
        except Exception as e:
            print(f"Error correcting text: {e}")
            import traceback
            traceback.print_exc()
            return text
    
    def correct_iterative(
        self, 
        text: str, 
        image_path: Optional[str] = None,
        verbose: bool = False
    ) -> Dict:
        """
        迭代修复（基于PPL反馈）
        
        Args:
            text: 待修复文本
            image_path: 可选的参考图像
            verbose: 是否打印详细信息
        
        Returns:
            result: 修复结果字典
        """
        current_text = text
        history = []
        
        # 计算原始PPL
        original_ppl = self.compute_perplexity([text], verbose=False)[0]
        current_ppl = original_ppl
        
        if verbose:
            print(f"\n{'='*70}")
            if image_path:
                print(f"Image: {image_path}")
            print(f"Original text: {text}")
            print(f"Original PPL: {original_ppl:.2f}")
            print(f"Threshold: {self.ppl_threshold}")
            print(f"{'='*70}")
        
        # 检查是否需要修复
        if current_ppl < self.ppl_threshold:
            if verbose:
                print(f"✓ Text is clean (PPL={current_ppl:.2f} < {self.ppl_threshold}), skipping correction")
            
            return {
                'original_text': str(text),
                'final_text': str(text),
                'original_ppl': float(original_ppl),
                'final_ppl': float(current_ppl),
                'iterations': int(0),
                'history': [],
                'converged': bool(True),
                'total_ppl_reduction': float(0.0),
                'skipped': bool(True)
            }
        
        # 迭代修复
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{self.max_iterations}:")
                print(f"  Current PPL: {current_ppl:.2f}")
                print(f"  Current text: {current_text}")
            
            # 修复
            corrected_text = self.correct_text(current_text, image_path)
            
            # 计算新PPL
            new_ppl = self.compute_perplexity([corrected_text], verbose=False)[0]
            
            # 记录历史
            history.append({
                'iteration': iteration + 1,
                'before_text': current_text,
                'after_text': corrected_text,
                'before_ppl': float(current_ppl),
                'after_ppl': float(new_ppl),
                'ppl_reduction': float(current_ppl - new_ppl)
            })
            
            if verbose:
                print(f"  → Corrected: {corrected_text}")
                print(f"  → New PPL: {new_ppl:.2f}")
                print(f"  → Reduction: {current_ppl - new_ppl:.2f}")
            
            # 检查收敛
            if new_ppl < self.ppl_threshold:
                if verbose:
                    print(f"  ✓ Converged! PPL={new_ppl:.2f} < {self.ppl_threshold}")
                current_text = corrected_text
                current_ppl = new_ppl
                break
            
            # 检查改进
            if corrected_text.strip() == current_text.strip():
                if verbose:
                    print(f"  ⚠ No change in text, stopping")
                break
            
            if new_ppl >= current_ppl - 1.0:
                if verbose:
                    print(f"  ⚠ No significant PPL improvement, stopping")
                break
            
            # 更新
            current_text = corrected_text
            current_ppl = new_ppl
        
        result = {
            'original_text': text,
            'final_text': current_text,
            'original_ppl': float(original_ppl),
            'final_ppl': float(current_ppl),
            'iterations': len(history),
            'history': history,
            'converged': bool(current_ppl < self.ppl_threshold),
            'total_ppl_reduction': float(original_ppl - current_ppl),
            'skipped': False
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Final Results:")
            print(f"  Original: {text} (PPL={original_ppl:.2f})")
            print(f"  Final:    {current_text} (PPL={current_ppl:.2f})")
            print(f"  Iterations: {len(history)}")
            print(f"  Converged: {'Yes ✓' if result['converged'] else 'No ✗'}")
            print(f"  Total PPL reduction: {original_ppl - current_ppl:.2f}")
            print(f"{'='*70}\n")
        
        return result
    
    def correct_batch_iterative(
        self, 
        texts: List[str],
        image_paths: Optional[List[str]] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        批量迭代修复
        
        Args:
            texts: 文本列表
            image_paths: 可选的图像路径列表
            verbose: 是否打印详细信息
        
        Returns:
            results: 修复结果列表
        """
        if image_paths and len(image_paths) != len(texts):
            raise ValueError("image_paths and texts must have same length")
        
        results = []
        
        # 首先批量计算所有文本的PPL
        print(f"\n{'='*70}")
        print(f"Batch Iterative Correction")
        print(f"Total samples: {len(texts)}")
        print(f"{'='*70}\n")
        
        ppls = self.compute_perplexity(texts, verbose=True)
        
        # 统计需要修复的样本
        needs_correction = ppls > self.ppl_threshold
        num_needs_correction = needs_correction.sum()
        
        print(f"\n✓ PPL Analysis Complete")
        print(f"  Clean samples (PPL < {self.ppl_threshold}): {len(texts) - num_needs_correction}")
        print(f"  Needs correction (PPL >= {self.ppl_threshold}): {num_needs_correction}")
        
        if num_needs_correction == 0:
            print(f"\n✓ All samples are clean!")
            for i, text in enumerate(texts):
                results.append({
                    'original_text': str(text),
                    'final_text': str(text),
                    'original_ppl': float(ppls[i]),
                    'final_ppl': float(ppls[i]),
                    'iterations': int(0),
                    'history': [],
                    'converged': bool(True),
                    'total_ppl_reduction': float(0.0),
                    'skipped': bool(True)
                })
            return results
        
        print(f"\nProcessing samples...")
        
        corrected_count = 0
        for i, text in enumerate(texts):
            img_path = image_paths[i] if image_paths else None
            
            if needs_correction[i]:
                corrected_count += 1
                
                if verbose or (corrected_count % 10 == 0):
                    print(f"\n[{corrected_count}/{num_needs_correction}] Processing sample {i+1}...")
                
                result = self.correct_iterative(
                    text, 
                    img_path, 
                    verbose=verbose
                )
            else:
                # 不需要修复
                result = {
                    'original_text': str(text),
                    'final_text': str(text),
                    'original_ppl': float(ppls[i]),
                    'final_ppl': float(ppls[i]),
                    'iterations': int(0),
                    'history': [],
                    'converged': bool(True),
                    'total_ppl_reduction': float(0.0),
                    'skipped': bool(True)
                }
            
            results.append(result)
        
        # 打印总体统计
        self._print_batch_statistics(results)
        
        return results
    
    def _print_batch_statistics(self, results: List[Dict]):
        """打印批量修复的统计信息"""
        print(f"\n{'='*70}")
        print("Batch Correction Statistics")
        print(f"{'='*70}")
        
        total = len(results)
        skipped = sum(1 for r in results if r.get('skipped', False))
        processed = total - skipped
        converged = sum(1 for r in results if r['converged'] and not r.get('skipped', False))
        
        print(f"Total samples: {total}")
        print(f"Skipped (already clean): {skipped} ({skipped/total*100:.1f}%)")
        print(f"Processed: {processed} ({processed/total*100:.1f}%)")
        
        if processed > 0:
            print(f"Converged: {converged}/{processed} ({converged/processed*100:.1f}%)")
            
            avg_iterations = np.mean([r['iterations'] for r in results if not r.get('skipped', False)])
            avg_ppl_reduction = np.mean([r['total_ppl_reduction'] for r in results if not r.get('skipped', False)])
            
            print(f"Average iterations: {avg_iterations:.2f}")
            print(f"Average PPL reduction: {avg_ppl_reduction:.2f}")
        
        # PPL分布
        original_ppls = [r['original_ppl'] for r in results if not np.isnan(r['original_ppl'])]
        final_ppls = [r['final_ppl'] for r in results if not np.isnan(r['final_ppl'])]
        
        print(f"\nPPL Statistics:")
        print(f"  Original - Mean: {np.mean(original_ppls):.2f}, Median: {np.median(original_ppls):.2f}")
        print(f"  Final    - Mean: {np.mean(final_ppls):.2f}, Median: {np.median(final_ppls):.2f}")
        print(f"{'='*70}\n")


def demo():
    """演示示例"""
    print("="*70)
    print("Unified VLM Corrector Demo")
    print("="*70)
    
    # 使用Qwen2-VL-2B（小模型，支持图像）
    corrector = UnifiedVLMCorrector(
        model_path="Qwen/Qwen2-VL-2B-Instruct",  # 或 "Qwen/Qwen2.5-3B-Instruct"
        ppl_threshold=50.0,
        max_iterations=3
    )
    
    # 测试样本
    test_texts = [
        "$!ow a dog sleeping wint a NDppy",
        "remove multiple p1ns and cahnge the background to gray",
        "This is a clean sentence with no errors."
    ]
    
    print("\n" + "="*70)
    print("Testing individual corrections")
    print("="*70)
    
    for text in test_texts:
        result = corrector.correct_iterative(text, verbose=True)


if __name__ == "__main__":
    demo()

