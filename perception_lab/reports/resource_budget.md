# 资源预算：估计与实测分开

初检 RTX 4090 24564MiB，驱动535.183.01，系统盘空闲约74GB，RAM62GiB且多数已用，swap2GiB已满。详见 logs/preflight_initial.json。nvcc11.8/12.2均已发现，驱动显示的CUDA12.2只是兼容上限。

当前准备预算：mini归档约4.17GB，解压预留8GB；COCO val归档约0.82GB，解压约0.82GB，val标注约20MB；2D与LiDAR权重先预留1GB；隔离环境复用已有torch，只新增兼容包，预留4GB；当前诊断预测/图像预留2GB。下载与解压期间保留归档，完成后仅本任务可重建缓存可按需清理。

全量trainval、地图、SurroundOcc、模型infos、正式预测和两个训练checkpoint组的实际预算须在数据清单到位后计算。当前可用空间不能视为足以容纳完整实验。没有正式benchmark前，不估算或承诺全流程完成时间。

性能：外部GPU任务存在时，不把混跑耗时称为独占warm p50/p95，也不报告TensorRT收益。CPU示例推理包含绘图，不是性能benchmark。
