# 固定教学案例索引

规则见 [teaching_cases.json](teaching_cases.json)。红圈为后验匹配 FP，橙叉为 FN；完整 token 与身份事件均保留在 JSON。这里是阈值教学诊断，不是官方 IDS。

| Split | 原因 | Frame | Sample token | GRAE | OptiPMB |
|---|---|---:|---|---|---|
| scene0061 | fp_improvement | 8 | `be99ffc878b24aca8956bbb4e0f97d0c` | [图片](scene0061_fp_improvement_grae_08.jpg) | [图片](scene0061_fp_improvement_optipmb_08.jpg) |
| scene0061 | fp_degradation | 9 | `9813c23a5f1448b09bb7910fea9baf20` | [图片](scene0061_fp_degradation_grae_09.jpg) | [图片](scene0061_fp_degradation_optipmb_09.jpg) |
| scene0061 | fn_degradation | 0 | `ca9a282c9e77460f8360f564131a8af5` | [图片](scene0061_fn_degradation_grae_00.jpg) | [图片](scene0061_fn_degradation_optipmb_00.jpg) |
| scene0061 | id_switch_improvement | 17 | `2ff86dc19c4644a1a88ce5ba848f56e5` | [图片](scene0061_id_switch_improvement_grae_17.jpg) | [图片](scene0061_id_switch_improvement_optipmb_17.jpg) |
| scene0061 | id_switch_degradation | 3 | `e0845f5322254dafadbbed75aaa07969` | [图片](scene0061_id_switch_degradation_grae_03.jpg) | [图片](scene0061_id_switch_degradation_optipmb_03.jpg) |
| mini_val | fp_improvement | 29 | `6d9984d09d52479e837da2fd09e192cc` | [图片](mini_val_fp_improvement_grae_29.jpg) | [图片](mini_val_fp_improvement_optipmb_29.jpg) |
| mini_val | fp_degradation | 48 | `5b7cb170eee6468aa1fdbd3abcf63c5a` | [图片](mini_val_fp_degradation_grae_48.jpg) | [图片](mini_val_fp_degradation_optipmb_48.jpg) |
| mini_val | fn_improvement | 52 | `372725a4b00e49c78d6d0b1c4a38b6e0` | [图片](mini_val_fn_improvement_grae_52.jpg) | [图片](mini_val_fn_improvement_optipmb_52.jpg) |
| mini_val | fn_degradation | 40 | `b5989651183643369174912bc5641d3b` | [图片](mini_val_fn_degradation_grae_40.jpg) | [图片](mini_val_fn_degradation_optipmb_40.jpg) |
| mini_val | id_switch_improvement | 2 | `c5f58c19249d4137ae063b0e9ecd8b8e` | [图片](mini_val_id_switch_improvement_grae_02.jpg) | [图片](mini_val_id_switch_improvement_optipmb_02.jpg) |
| mini_val | id_switch_degradation | 1 | `3950bd41f74548429c0f7700ff3d8269` | [图片](mini_val_id_switch_degradation_grae_01.jpg) | [图片](mini_val_id_switch_degradation_optipmb_01.jpg) |
