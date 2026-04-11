import json
import os
import sys
import argparse

# ================= 配置区域 =================
# 请确保这里指向你的 CIRR captions 目录
# 或者是你 config.py 里定义的 CIRR_ANNOTATION_DIR
ANNOTATION_DIR = '/cirr_dataset/cirr/captions' 

# 引入你的 corrupt 库
# 如果 corrupt 文件夹在当前目录下，直接 import
# 如果在子文件夹，请 sys.path.append
try:
    from corrupt import text_corrupt as txt_crpt
    print("✅ 成功导入 corrupt 库")
except ImportError:
    # 尝试添加路径 (根据你上传的文件夹结构)
    sys.path.append("Benchmark-Robustness-Text-Image-Compose-Retrieval") 
    try:
        from corrupt import text_corrupt as txt_crpt
        print("✅ 成功导入 corrupt 库 (通过路径添加)")
    except ImportError:
        print("❌ 错误: 找不到 corrupt 库，请检查路径")
        sys.exit(1)
# ===========================================

def generate_corrupted_file(noise_type, severity=3, split='test1'):
    original_file = f'cap.rc2.{split}.json'
    original_path = os.path.join(ANNOTATION_DIR, original_file)
    
    output_filename = f'cap.rc2.{split}_{noise_type}.json'
    output_path = os.path.join(ANNOTATION_DIR, output_filename)

    if not os.path.exists(original_path):
        print(f"❌ 找不到原始文件: {original_path}")
        return

    print(f"📖 正在读取: {original_file}")
    with open(original_path, 'r') as f:
        data = json.load(f)

    print(f"⚡ 正在应用噪声: {noise_type} (Severity: {severity})...")
    corrupt_func = getattr(txt_crpt, noise_type)

    new_data = []
    for item in data:
        new_item = item.copy()
        original_caption = item['caption']
        
        try:
            # 某些函数返回 (text, dist)，有的返回 text
            res = corrupt_func(original_caption, severity)
            if isinstance(res, tuple) or isinstance(res, list):
                new_caption = res[0]
            else:
                new_caption = res
        except Exception as e:
            # 如果报错（比如空字符串），保持原样
            new_caption = original_caption

        new_item['caption'] = new_caption
        new_data.append(new_item)

    with open(output_path, 'w') as f:
        json.dump(new_data, f)
    
    print(f"💾 已保存: {output_filename} (包含 {len(new_data)} 条数据)")

def main():
    # 定义你要生成的噪声列表（建议选这 3 个代表性的）
    NOISE_TYPES = [
        'misspelling_filter',  # 拼写错误 (Type A)
        'qwerty_filter',       # 键盘误触 (Type B)
        'RemoveChar_filter',   # 字符缺失 (Type C - 强破坏)
    ]

    for noise in NOISE_TYPES:
        generate_corrupted_file(noise, severity=3) # 也可以改成 5 测试极限

if __name__ == '__main__':
    main()