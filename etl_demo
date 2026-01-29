import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import load_dataset
from PIL import Image

# ==========================================
# 1. 辅助函数：坐标转换与归一化
# ==========================================
def normalize_bbox(coco_bbox, width, height):
    """
    将 COCO 格式 [x, y, w, h] 转换为 归一化 [x_min, y_min, x_max, y_max]
    范围 0.0 - 1.0
    """
    x, y, w, h = coco_bbox
    
    # 防止越界 (有些数据集标注可能会超出图片边缘)
    x_min = max(0, x) / width
    y_min = max(0, y) / height
    x_max = min(width, x + w) / width
    y_max = min(height, y + h) / height
    
    # 保留4位小数，节省空间且足够精确
    return [round(v, 4) for v in [x_min, y_min, x_max, y_max]]

# ==========================================
# 2. 核心逻辑：ETL (Extract, Transform, Load)
# ==========================================
def run_etl_pipeline(num_samples=5):
    print(f"🚀 开始流式读取 Visual Genome 数据 (只取前 {num_samples} 条)...")
    
    # 使用 streaming=True，无需下载整个数据集，秒级启动
    # region_descriptions_v1.2.0 包含图片区域描述和 bbox
    dataset = load_dataset("visual_genome", "region_descriptions_v1.2.0", split="train", streaming=True)
    
    unified_data_list = []
    
    # 这里的 iterator 会从网络流式获取数据
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
            
        # 原始数据提取
        # Visual Genome 的 HF 格式通常包含: image (PIL对象), regions (列表)
        image = item['image']
        width, height = image.size
        regions = item['regions'] # 这是一个 list，里面有很多个 bbox 和 phrase
        
        # --- 构建统一 Schema ---
        sample_entry = {
            "id": f"vg_{item['image_id']}",
            "data_source": "visual_genome",
            "task_type": "spatial_understanding",
            "media": {
                "image_size": [width, height],
                "image_path": f"virtual_path/vg_{item['image_id']}.jpg" # 模拟路径
            },
            "spatial_annotations": [],
            "conversations": []
        }
        
        # 处理该图片内的所有标注区域 (这里只取前3个做演示，避免过长)
        for region in regions[:3]:
            # 原始 bbox 是 [x, y, w, h]
            raw_bbox = [region['x'], region['y'], region['width'], region['height']]
            norm_bbox = normalize_bbox(raw_bbox, width, height)
            phrase = region['phrase']
            
            # 填充 Annotation
            sample_entry["spatial_annotations"].append({
                "label": "region_description",
                "bbox_2d": norm_bbox,
                "text": phrase
            })
            
            # 填充 Conversation (构造指令微调格式)
            # 模拟用户问：这个区域是什么？
            # 模拟 AI 答：描述 + 坐标
            sample_entry["conversations"].append({
                "from": "human",
                "value": f"Describe the region at <box>{norm_bbox}</box>."
            })
            sample_entry["conversations"].append({
                "from": "gpt",
                "value": phrase
            })
            
        unified_data_list.append(sample_entry)
        
        # --- 3. 实时可视化验证 (只画第一张图做证明) ---
        if i == 0:
            visualize_verification(image, sample_entry)

    # 保存结果
    output_file = "unified_spatial_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in unified_data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ 处理完成！数据已保存至 {output_file}")
    print("你可以打开这个文件查看格式，或者展示生成的 'verification_plot.png' 给面试官。")

# ==========================================
# 3. 可视化模块 (Proof of Work)
# ==========================================
def visualize_verification(image, schema_entry):
    print("🎨 正在生成可视化验证图...")
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    img_w, img_h = image.size
    
    # 从我们转化好的 Schema 里读数据，反向画回去，证明转化无误
    for ann in schema_entry["spatial_annotations"]:
        # 拿到归一化坐标 [x1, y1, x2, y2]
        nx1, ny1, nx2, ny2 = ann["bbox_2d"]
        
        # 反归一化回像素坐标
        x = nx1 * img_w
        y = ny1 * img_h
        w = (nx2 - nx1) * img_w
        h = (ny2 - ny1) * img_h
        
        # 画框
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # 写字 (防止遮挡，写在框上方)
        plt.text(x, y - 5, ann["text"], color='white', fontsize=10, 
                 bbox=dict(facecolor='red', alpha=0.5))
        
    plt.axis('off')
    plt.title(f"Verification: {schema_entry['id']} (Normalized BBoxes restored)")
    plt.savefig("verification_plot.png")
    plt.show()

if __name__ == "__main__":
    run_etl_pipeline(num_samples=5)
