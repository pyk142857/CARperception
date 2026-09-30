# 感知流水线耗时测量

目标：实测各模块和离线阶段耗时，区分首次初始化、推理API、后处理/导出、跟踪和Rerun构建。

执行入口：`tools/profile_perception_pipeline.py`。本轮顺序运行13个独立任务，前3个时刻六相机用于各模型，跟踪及记录导出单独使用既有39帧。视觉模型显式采用CUDA，不沿用历史CPU日志。所有GPU阶段边界同步，输入和输出在独立计时目录。

补充：`--resident-probes`将3D模型常驻，首帧预热3次后重复20次，分别测CenterPoint、PointPillars；不混入主任务总占比。相机模型另报告去除首次调用后的参考值。

验收：13主任务与2常驻探针均退出0；阶段时间非负、阶段和与墙钟残差明确记录；日志、计时JSON、源码哈希、GPU信息上传。不得将不同帧数的任务占比称为生产实时占比，不得将API时长当作裸网络forward；不覆盖当前结果与回放。

已完成。见[报告](../reports/pipeline_timing/report.md)和[验证](../reports/pipeline_timing/verification.json)。
