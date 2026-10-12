# 复现与运行记录

本轮已运行命令、捕获路径、环境、版本及哈希分别记录于：

- [reference 重复](reference/run_commands.md)、[同输入回放](reference/replay/execution.json)
- [TRT 构建、重复、回放](trt/run_commands.md)、[代码版本限制](trt/runner_version_provenance.json)
- [P4 CPU 审计](p4_audit/audit.py)、[验证日志](p4_audit/verification.log)
- [发布批次与逐文件校验](publication_transport.json)、[远端回执](publication_receipt.json)

GPU 实验顺序执行，保持生产库与 engine，不把诊断同步/文件写出耗时当生产时延。原始捕获在 `perception_lab/runs/deployment_4090_20261012_followup/{reference,trt}/`，不进入 Git。生成的大 engine 使用原权重和工具重建，来源身份见 `trt/artifact_identity.json`。

在项目根目录运行最终代码验证：

```bash
perception_lab/envs/bevfusion/bin/python -m pytest -q perception_lab/tests/test_bevfusion_reference_followup.py perception_lab/tests/test_deployment_runner.py perception_lab/tests/test_bevfusion_4090.py perception_lab/tests/test_sparse4d_4090.py perception_lab/tests/test_bev_pooling_provenance.py perception_lab/tests/test_detection_export.py perception_lab/tests/test_camera_overlay.py perception_lab/tests/test_geometry.py
```

publication 使用独立 `/home/minglei/Desktop/CAR-publish-20261012` worktree，普通提交/推送，命令级 `git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 push`，未修改全局配置。网络失败时重试，GitHub API 获取实际 blob 内容校验，认证不进入产物。首次恢复完成 main 为 `1f866fc8c493588b722902cdd0e503c1cd5daedb`；本次补查发布见回执。
