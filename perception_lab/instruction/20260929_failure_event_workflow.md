# Failure event workflow — design and implementation plan

Goal: 实现用户已确认的“原始案例 → 目标事件 → 问题分组 → 复核状态 → Rerun 片段回放”，限 mini 教学闭环。
Architecture: 离线确定性归并生成 events.json；本机 9092 提供事件、复核 API 与现有 WebViewer。原始 cases、评估和 RRD 不改写。
Tech stack: Python / NumPy / SciPy（现有 mmdet3d 环境），stdlib HTTP + 原子 JSON，原生 JS / Rerun 0.23.4。

## 规则与验收
- 有真值：scene/module/kind/class/instance 分组，严格相邻帧且间隔≤0.75秒；ID 切换仍保存每次转换。
- FP：scene/module/class 分组，统一全局坐标；上一帧速度预测后 gated Hungarian，残差≤2米、尺寸比≤2；tracking 额外要求预测 ID 相同。无跨空帧补接。
- FP 关联是启发式候选，不表示真值身份。没有相同 ID 的跟踪 FP 不强行跨 ID 合并。
- 每条有效失败只属于一个事件；gap_recovery 排除；稳定 ID、首末时间、代表帧、原始 case_ids、转换序列、关联方法可追溯。
- 问题分组为 module/kind/class/事件中位距离桶（0–20、20–40、40+米）；只是现象分类，不作因果判断；无曝光量，不生成失败率。
- 复核目标支持事件与分组，状态 pending/confirmed/not_issue/annotation_error/duplicate/investigating/resolved；负责人、根因、措施、验证说明和重复目标；resolved 必须有验证说明，duplicate 必须指定有效且不同的目标。
- JSON 乐观版本号、线程锁、原子替换；数据版本校验；历史记录；刷新保持；导出快照。不直接写模型或评估。
- 播放使用真实 elapsed 时间戳选择前后2秒，同场景内裁剪，约2fps，结束自动暂停；取消/新选择终止旧片段。
- 默认事件列表；分组榜按事件数量排序；原始案例入口与旧 case 链接保留。复核可在 RRD 未完成加载前操作。

## 实施步骤（当前会话直接执行）
- [x] tests/test_failure_events.py：先覆盖间断/场景隔离/全局坐标补偿/FP一对一/ID转换/稳定性与原始记录守恒，再运行确认未实现。
- [x] tools/aggregate_failure_events.py：实现归并与确定性JSON、来源哈希、统计输出；执行真实数据导出并验证1910条一一覆盖。
- [x] tools/event_reviews.py + tests/test_event_reviews.py：先验证持久化、冲突、状态校验、旧数据隔离，再接入 start_case_browser.py API。
- [x] web/case_browser/event_ui.js + event_logic.mjs：事件/分组/案例视图、复核表单、展开案例、前后片段控制；app.js 保持原Viewer加载逻辑。
- [x] tests/test_event_logic.mjs：时间戳窗口、首尾裁剪与播放停止；Playwright实际验证分组→事件→案例→跳帧→片段结束→复核持久化。
- [x] reports/failure_events：结果、参数、局限、验证JSON、截图；更新README/RERUN/TASK_STATE；全套检查后提交推送main，核验远端。

复现核心命令：
```bash
perception_lab/envs/mmdet3d/bin/python -m unittest discover -s perception_lab/tests
perception_lab/envs/mmdet3d/bin/python perception_lab/tools/aggregate_failure_events.py
node --experimental-modules perception_lab/tests/test_event_logic.mjs
perception_lab/envs/rerun/bin/python perception_lab/tools/start_case_browser.py --no-browser
```
