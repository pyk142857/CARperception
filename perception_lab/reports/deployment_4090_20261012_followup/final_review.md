# 最终只读审阅（2026-10-12）

审阅对象为总report/status/metrics/run_commands、reference与TRT子报告、最新reference/replay结果、P4及已完成独立审计。未使用GPU，未修改实现和旧报告。

**未发现阻塞发布的结论或数值矛盾。** 总状态保持partial；原全候选一致性失败、官方mini精度预算通过、无新精度修复、pooling无稳定提速、GaussianWorld评估blocked_data均分别表述。原39/81帧评估复用与本轮4诊断帧范围没有混淆。marked图候选来源不冒充生产来源；AA行顺序不冒充对象身份；非插桩与插桩差异不直接作为hook因果归因。

核对40次参考共同输入回放、8次映射对照、144行局部指标。非head全部可比输出通过固定容差；head来源公共数为5组200、3组199，reg/height/dim/rot/vel的超容差范围分别0–3/0–4/0–10/0–4/0–5，score均0，与子报告一致。8次映射对照的排序控制均逐位相同，重建参考geometry匹配捕获记录。

额外直接读取两组geometry NPZ、核验文件SHA256并重新计算完整张量统计：747/reference上游为25个超容差、max=0.1086292267；e084/reference上游为96个超容差、max=0.2268255055，与保存结果逐项相同。总报告几何格点差异126/104/136/156的token顺序已明确，对应正确。其余大型模块数组沿用本轮已完成独立审计，不声称重复执行GPU验证。

最终execution exit_code=0、40模块/8映射数量、runner实际SHA256与summary/execution一致。final_tests.log实际包含22 passed in 0.76s；本审阅核对日志，没有再次运行测试。小范围抽查主metrics与原官方摘要、pooling负收益、Sparse4D及GT计数一致。

发布验收属于主任务：原实验main恢复已有回执，本次新诊断的最终main与逐文件远端验证须在发布后写入publication_receipt.json。本审阅不将本地审核当作新诊断已上传。
