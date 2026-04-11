"""
测试防御流程安装和设置
快速验证所有组件是否正常工作
"""

import sys
from pathlib import Path

def test_imports():
    """测试必要的库是否安装"""
    print("="*70)
    print("Testing Imports...")
    print("="*70)
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name:15s} - OK")
        except ImportError:
            print(f"✗ {name:15s} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required packages installed!")
    return True


def test_cuda():
    """测试CUDA是否可用"""
    print("\n" + "="*70)
    print("Testing CUDA...")
    print("="*70)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print(f"✗ CUDA not available (will use CPU - slower)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def test_modules():
    """测试自定义模块是否可以导入"""
    print("\n" + "="*70)
    print("Testing Custom Modules...")
    print("="*70)
    
    modules = [
        'ppl_detector',
        'text_corrector',
        'cirr_defense'
    ]
    
    success = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module:20s} - OK")
        except ImportError as e:
            print(f"✗ {module:20s} - FAILED: {e}")
            success = False
    
    if success:
        print("\n✓ All custom modules loaded successfully!")
    else:
        print("\n⚠️  Some modules failed to load. Check file paths.")
    
    return success


def test_data_path():
    """测试CIRR数据路径"""
    print("\n" + "="*70)
    print("Testing Data Paths...")
    print("="*70)
    
    base_path = Path(__file__).resolve().parent.parent
    cirr_path = base_path / 'cirr_dataset' / 'cirr' / 'captions'
    
    print(f"Base path: {base_path}")
    print(f"CIRR captions path: {cirr_path}")
    
    if not cirr_path.exists():
        print(f"\n✗ CIRR caption path not found!")
        print(f"  Expected: {cirr_path}")
        return False
    
    # 检查文件
    files = ['cap.rc2.test1.json', 'cap.rc2.train.json', 'cap.rc2.val.json']
    missing = []
    
    for fname in files:
        fpath = cirr_path / fname
        if fpath.exists():
            print(f"✓ {fname:25s} - Found")
        else:
            print(f"✗ {fname:25s} - Missing")
            missing.append(fname)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    print("\n✓ All CIRR data files found!")
    return True


def test_ppl_detector():
    """测试PPL检测器"""
    print("\n" + "="*70)
    print("Testing PPL Detector...")
    print("="*70)
    
    try:
        from ppl_detector import PPLDetector
        
        print("Initializing PPL Detector (this may take a moment)...")
        detector = PPLDetector(model_name='gpt2', device='cuda')
        
        # 测试样本
        test_texts = [
            "Put a clock on the wall",  # 正常
            "cahnge the background"      # 拼写错误
        ]
        
        print("\nComputing PPL for test samples...")
        ppls = detector.compute_perplexity(test_texts)
        
        print(f"\nResults:")
        for text, ppl in zip(test_texts, ppls):
            print(f"  PPL={ppl:6.1f} : {text}")
        
        if ppls[1] > ppls[0]:
            print(f"\n✓ PPL detector working correctly!")
            print(f"  (Error text has higher PPL: {ppls[1]:.1f} > {ppls[0]:.1f})")
            return True
        else:
            print(f"\n⚠️  Unexpected PPL values")
            return False
            
    except Exception as e:
        print(f"\n✗ Error testing PPL detector: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*70)
    print("CIRR Defense Pipeline - Installation Test")
    print("#"*70 + "\n")
    
    results = {
        'Imports': test_imports(),
        'CUDA': test_cuda(),
        'Modules': test_modules(),
        'Data Paths': test_data_path(),
    }
    
    # PPL检测器测试（可选，因为需要下载模型）
    print("\n" + "="*70)
    test_ppl = input("Test PPL Detector? (This will download GPT-2 model ~500MB) [y/N]: ")
    if test_ppl.lower() == 'y':
        results['PPL Detector'] = test_ppl_detector()
    else:
        print("Skipping PPL Detector test")
        results['PPL Detector'] = None
    
    # 总结
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{test_name:20s} : {status}")
    
    all_critical_passed = all(v for k, v in results.items() if k != 'PPL Detector' and v is not None)
    
    print("\n" + "="*70)
    if all_critical_passed:
        print("✅ Setup looks good! You can proceed with:")
        print("   python cirr_defense.py --split test1")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()




