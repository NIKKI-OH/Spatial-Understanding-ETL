Unified Spatial Understanding Data Pipeline 🚀
这是一个针对空间理解任务（Spatial Understanding）的多模态数据统一处理工程。

🎯 项目目标
针对当前开源数据集格式不统一的痛点，设计了一套通用的 Instruction-Tuning (指令微调) 格式，并完成了从原始数据到统一 Schema 的 ETL 转换流程。

🛠️ 支持的数据集
调研并统一了以下三种典型任务的数据格式：

VQA / Spatial Reasoning: Visual Genome (基于区域的密集描述)

Robotics / Trajectory: Open X-Embodiment (机械臂轨迹) [WIP]

3D / Affordance: ScanNet (3D 语义分割) [WIP]

📊 统一 Schema 设计
设计了包含 conversations (对话) 与 spatial_annotations (结构化标注) 的双层架构：
{
  "id": "vg_1059",
  "task_type": "spatial_understanding",
  "conversations": [
    {"from": "human", "value": "Describe the region..."},
    {"from": "gpt", "value": "A white cat sitting on the sofa."}
  ],
  "spatial_annotations": [
    {"bbox_2d": [0.1, 0.2, 0.5, 0.6], "label": "cat"}
  ]
}
