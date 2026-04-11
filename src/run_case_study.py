import argparse
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from data_utils import targetpad_transform

def analyze_vmf_uncertainty(model, image_path, texts, txt_processors, device):
    """全方位提取模型内部的 vMF 浓度参数 (Kappa)，追踪纯粹数学约束下的不确定性"""
    model.eval()
    
    # 1. 处理图片
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ 无法打开图片: {image_path}. 错误: {e}")
        return

    preprocess = targetpad_transform(1.25, 224)
    image = preprocess(img).unsqueeze(0).to(device)
    
    print(f"\n📸 测试图片: {image_path}")
    print("=" * 90)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            
            # ---------------------------------------------------------
            # 👁️ 步骤 1: 提取纯视觉不确定性 κ_v (即目标图像的 kappa)
            # 注意: 新模型的 target 返回的是 32 个 token 的特征，但 kappa 仍是标量
            # ---------------------------------------------------------
            target_feats, img_kappas = model.extract_target_features(image)
            k_v = img_kappas.squeeze().item()
            print(f"🖼️  [参考图像置信度 κ_v]: {k_v:>6.2f} (反映图像质量与主体清晰度)")
            print("-" * 90)

            # ---------------------------------------------------------
            # 📝 步骤 2 & 3: 提取纯文本与融合特征的不确定性
            # ---------------------------------------------------------
            for raw_text in texts:
                text = txt_processors["eval"](raw_text)
                
                # 🚀 核心适配: 直接调用你新模型里的 extract_query_debug_features
                # 它会一次性返回 k_q_raw, k_q_final, k_t, k_v
                debug_info = model.extract_query_debug_features(image, text)
                
                k_t = debug_info["k_t"].item()
                k_q_raw = debug_info["k_q_raw"].item()
                k_q_final = debug_info["k_q_final"].item()  # 包含动态融合衰减的最终置信度
                
                # 计算理论期望衰减 A_d (基于最终的 κ_q_final)
                feature_dim = model.embed_dim  # 直接从模型获取维度 (默认 256)
                nu = feature_dim / 2.0 - 1.0
                a_d = k_q_final / (k_q_final + nu)
                
                # ---------------------------------------------------------
                # 📊 打印溯源对比
                # ---------------------------------------------------------
                print(f"📝 文本指令: '{raw_text}'")
                print(f"   💬 [纯文本置信度 κ_t]: {k_t:>6.2f}")
                print(f"   🧬 [原始融合置信度 κ_q_raw]: {k_q_raw:>6.2f}")
                print(f"   🔥 [动态融合终态 κ_q_final]: {k_q_final:>6.2f} | [期望衰减系数 A_d]: {a_d:>6.4f}")
                
                # 💡 动态溯源逻辑分析 (使用动态融合后的 k_q_final 作为主要判据)
                if k_q_final < 2.5: 
                    if k_v > 5.0 and k_t > 5.0:
                        status = "🚨 [跨模态冲突] 图文单看都清晰，但融合时产生语义断裂，或局部更新门控(Gate)严重偏离，特征被摊平！"
                    else:
                        status = "🚨 [单模态拖累] 文本指令错乱或图像太差，导致联合特征丧失方向判别力。"
                elif k_q_final < 12.0:
                    status = "🟡 [常规确信度] 正常的组合修改指令，置信度处于健康动态区间。"
                else:
                    status = "✅ [绝对确信] 图文对齐与局部更新机制(Top-K Token)极其完美，方向向量极其尖锐！"
                
                print(f"   💡 分析: {status}")
                print("-" * 90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="训练好的 .pth 权重路径")
    parser.add_argument("--image-path", type=str, required=True, help="参考图片路径")
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🤖 正在初始化模型架构...")
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=args.blip_model_name, model_type="pretrain", is_eval=True, device=device
    )
    blip_model = blip_model.to(device)
    
    print(f"💾 正在从 {args.model_path} 加载权重...")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    if 'model_state_dict' in state_dict: 
        state_dict = state_dict['model_state_dict']
    
    # 🔍 防丢权重“照妖镜”逻辑
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    msg = blip_model.load_state_dict(clean_state_dict, strict=False)
    
    kappa_missing = [k for k in msg.missing_keys if 'kappa' in k]
    if kappa_missing:
        print("\n" + "!"*80)
        print("🚨 致命警告：Kappa 预测器的权重丢失了！")
        for k in kappa_missing[:5]: 
            print(f"  - {k}")
        print("!"*80 + "\n")
    else:
        print("✅ 所有 Kappa 预测器权重加载成功，无丢失！\n")

    # 测试用例
    test_texts = [
        "remove all but one dog and add a woman hugging it", 
        "remvoe all but one dog and add a woman hugging it", # 轻微拼写错误
        "all but one dog and add a woman hugging it",        # 缺失动词
        "turn the dog into a flying airplane",               # 跨模态逻辑冲突
        "jfakldsjfioqwe asdf zxcvq"                          # 纯随机噪声
    ]
    
    analyze_vmf_uncertainty(blip_model, args.image_path, test_texts, txt_processors, device)