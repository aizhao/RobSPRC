"""
Test script for Unified VLM Corrector
测试统一VLM修复器
"""

from unified_vlm_corrector import UnifiedVLMCorrector


def test_basic_functionality():
    """测试基本功能"""
    print("="*70)
    print("Testing Unified VLM Corrector - Basic Functionality")
    print("="*70)
    
    # 初始化（使用较小的模型进行测试）
    print("\n1. Initializing model...")
    try:
        corrector = UnifiedVLMCorrector(
            model_path="Qwen/Qwen2.5-3B-Instruct",  # 先用纯文本模型测试
            ppl_threshold=50.0,
            max_iterations=3
        )
        print("✓ Model initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False
    
    # 测试PPL计算
    print("\n2. Testing PPL computation...")
    test_texts = [
        "This is a clean sentence.",
        "$!ow a dog sleeping wint a NDppy",  # 扰动文本
        "remove multiple p1ns and cahnge the background"  # 扰动文本
    ]
    
    try:
        ppls = corrector.compute_perplexity(test_texts, verbose=False)
        print("✓ PPL computation successful!")
        for text, ppl in zip(test_texts, ppls):
            status = "CLEAN" if ppl < corrector.ppl_threshold else "PERTURBED"
            print(f"  [{status}] PPL={ppl:.2f}: {text[:50]}")
    except Exception as e:
        print(f"✗ PPL computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试单次修复
    print("\n3. Testing single correction...")
    perturbed_text = "$!ow a dog sleeping wint a NDppy"
    
    try:
        corrected = corrector.correct_text(perturbed_text)
        print("✓ Correction successful!")
        print(f"  Original:  {perturbed_text}")
        print(f"  Corrected: {corrected}")
    except Exception as e:
        print(f"✗ Correction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试迭代修复
    print("\n4. Testing iterative correction...")
    
    try:
        result = corrector.correct_iterative(perturbed_text, verbose=True)
        print("✓ Iterative correction successful!")
    except Exception as e:
        print(f"✗ Iterative correction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    return True


def test_batch_correction():
    """测试批量修复"""
    print("\n" + "="*70)
    print("Testing Batch Correction")
    print("="*70)
    
    corrector = UnifiedVLMCorrector(
        model_path="Qwen/Qwen2.5-3B-Instruct",
        ppl_threshold=50.0,
        max_iterations=2
    )
    
    test_texts = [
        "This is a clean sentence.",  # 干净
        "Another normal text here.",  # 干净
        "$!ow a dog sleeping wint a NDppy",  # 扰动
        "remove multiple p1ns and cahnge the background",  # 扰动
        "the lunch room has a wite sofa"  # 扰动
    ]
    
    print(f"\nProcessing {len(test_texts)} samples...")
    
    try:
        results = corrector.correct_batch_iterative(test_texts, verbose=False)
        
        print("\n" + "="*70)
        print("Results:")
        print("="*70)
        
        for i, (text, result) in enumerate(zip(test_texts, results)):
            print(f"\n[{i+1}] Original:  {text}")
            print(f"    Final:     {result['final_text']}")
            print(f"    PPL: {result['original_ppl']:.2f} → {result['final_ppl']:.2f}")
            print(f"    Iterations: {result['iterations']}")
            print(f"    Status: {'Skipped (clean)' if result.get('skipped') else 'Corrected'}")
        
        print("\n✓ Batch correction successful!")
        return True
        
    except Exception as e:
        print(f"✗ Batch correction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Unified VLM Corrector - Test Suite")
    print("="*70 + "\n")
    
    # 测试基本功能
    if not test_basic_functionality():
        print("\n✗ Basic functionality tests failed!")
        return
    
    # 测试批量修复
    if not test_batch_correction():
        print("\n✗ Batch correction tests failed!")
        return
    
    print("\n" + "="*70)
    print("All Tests Completed Successfully! ✓")
    print("="*70)
    print("\nYou can now run the CIRR defense pipeline with:")
    print("  python cirr_defense_unified.py --split test1")


if __name__ == "__main__":
    main()




