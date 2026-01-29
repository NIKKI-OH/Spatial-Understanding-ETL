import json
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

# ==========================================
# 0. 准备全模态样本 (All-in-One Data)
# 我们手动为不同的图片分配不同的任务类型
# ==========================================
REAL_SAMPLES = [
    {
        # 任务 1: 物体检测 (Detection)
        "id": "task_bbox_cat",
        "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "task_type": "detection",
        "label": "cat",
        "data": {
            "bbox": [14, 3, 310, 477] # [x, y, w, h]
        },
        "instruction": "Detect the cat."
    },
    {
        # 任务 2: 轨迹预测 (Trajectory) - 模拟滑雪路径
        "id": "task_traj_skier",
        "url": "http://images.cocodataset.org/val2017/000000000785.jpg",
        "task_type": "trajectory",
        "label": "skier_path",
        "data": {
            # 模拟一串点：从头顶滑下来的轨迹 [[x,y], [x,y]...]
            "points": [[250, 20], [260, 100], [280, 200], [300, 300], [220, 350]]
        },
        "instruction": "Predict the future trajectory of the skier."
    },
    {
        # 任务 3: 可供性/操作点 (Affordance) - 模拟机器人应该看向哪里
        "id": "task_affordance_sign",
        "url": "http://images.cocodataset.org/val2017/000000000724.jpg",
        "task_type": "affordance",
        "label": "stop_sign_center",
        "data": {
            # 关注点/抓取点 [x, y]
            "point": [343, 202] 
        },
        "instruction": "Where is the center of the stop sign for interaction?"
    }
]

# ==========================================
# 1. 核心逻辑：通用 ETL 流水线
# ==========================================
def download_image(url):
    print(f"📥 下载中: {url} ...")
    try:
        response = requests.get(url, timeout=10)
        return Image.open(BytesIO(response.content))
    except:
        return None

def normalize_coords(coords, w, h, type="bbox"):
    """万能归一化函数：支持 bbox, point, trajectory"""
    if type == "bbox":
        x, y, bw, bh = coords
        return [round(x/w, 3), round(y/h, 3), round((x+bw)/w, 3), round((y+bh)/h, 3)]
    elif type == "point":
        return [round(coords[0]/w, 3), round(coords[1]/h, 3)]
    elif type == "trajectory":
        return [[round(p[0]/w, 3), round(p[1]/h, 3)] for p in coords]

def run_multimodal_pipeline():
    print("🚀 启动全模态空间理解流水线 (BBox + Traj + Affordance)...")
    
    unified_data = []
    
    for item in REAL_SAMPLES:
        image = download_image(item["url"])
        if not image: continue
        w, h = image.size
        
        # --- 1. 构建统一格式 (Unified Schema) ---
        entry = {
            "id": item["id"],
            "source": "coco_simulated",
            "task_type": item["task_type"],
            "media": {"image_size": [w, h], "url": item["url"]},
            "spatial_annotations": [],
            "conversations": []
        }
        
        # --- 2. 根据不同任务类型处理数据 ---
        raw_data = item["data"]
        
        if item["task_type"] == "detection":
            norm_box = normalize_coords(raw_data["bbox"], w, h, "bbox")
            entry["spatial_annotations"].append({
                "type": "bbox", "value": norm_box, "label": item["label"]
            })
            gpt_resp = f"Found at <box>{norm_box}</box>."

        elif item["task_type"] == "trajectory":
            norm_traj = normalize_coords(raw_data["points"], w, h, "trajectory")
            entry["spatial_annotations"].append({
                "type": "trajectory", "value": norm_traj, "label": item["label"]
            })
            gpt_resp = f"Trajectory path: <traj>{norm_traj}</traj>."

        elif item["task_type"] == "affordance":
            norm_point = normalize_coords(raw_data["point"], w, h, "point")
            entry["spatial_annotations"].append({
                "type": "point", "value": norm_point, "label": item["label"]
            })
            gpt_resp = f"Interact at point: <point>{norm_point}</point>."

        # 填入对话
        entry["conversations"] = [
            {"from": "human", "value": item["instruction"]},
            {"from": "gpt", "value": gpt_resp}
        ]
        
        unified_data.append(entry)
        
        # --- 3. 可视化验证 (画出不同的形状) ---
        visualize_task(image, item, f"verify_{item['task_type']}.png")

    # 保存 JSONL
    with open("unified_multimodal_data.jsonl", "w") as f:
        for d in unified_data:
            f.write(json.dumps(d) + "\n")
    print("✅ 全模态数据处理完成！")

# ==========================================
# 2. 可视化模块 (根据任务画不同的图)
# ==========================================
def visualize_task(image, item, save_name):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    data = item["data"]
    task = item["task_type"]
    
    if task == "detection":
        # 画框
        x, y, w, h = data["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='#00FF00', facecolor='none')
        ax.add_patch(rect)
        plt.title(f"Task: Detection (BBox) - {item['label']}")
        
    elif task == "trajectory":
        # 画线 (轨迹)
        points = data["points"]
        # 解压 x 和 y 坐标列表
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # 画红色的轨迹线，带点
        plt.plot(xs, ys, color='red', linewidth=4, marker='o', markersize=8)
        plt.title(f"Task: Trajectory (Path) - {item['label']}")
        
    elif task == "affordance":
        # 画点 (热力点/操作点)
        x, y = data["point"]
        # 画一个半透明的圆
        circle = patches.Circle((x, y), radius=20, color='blue', alpha=0.6)
        ax.add_patch(circle)
        # 画中心十字
        plt.plot(x, y, 'w+', markersize=10)
        plt.title(f"Task: Affordance (Interaction Point) - {item['label']}")

    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_multimodal_pipeline()
