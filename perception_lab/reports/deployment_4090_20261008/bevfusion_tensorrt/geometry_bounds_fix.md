# 修复 CUDA geometry 最后区间遗漏点

真实首帧的PyTorch有效投影贡献为1,835,478，CUDA interval累计为1,835,477。定位到上游 `collect_starts_kernel`：最后区间终点设为 `numel_geometry-1`，但下游按 `i<interval.y` 使用排他终点，最后一个排序后的有效点始终遗漏。

修复为 `numel_geometry`，其余索引、排序、特征和累加规则不变。独立补丁 [bevfusion_geometry_bounds.patch](../../../tools/bevfusion_geometry_bounds.patch)。幂等复现包含在 `tools/bevfusion_trt_setup.sh` 中，与自主pooling候选补丁分开。

这是geometry正确性修复，不称为pooling算子提速；上游pooling与候选pooling都使用同样修复后的intervals。前后真实推理和计时应分别记录，不能拿旧capture覆盖新构建身份。

实际生产CUDA kernel回归测试见 `tests/bevfusion_geometry_bounds_test.cu`：直接包含被测camera-geometry.cu并启动原 `collect_starts_kernel`，不是用Python重写公式自我验证。相同测试分别针对修复前文件和当前源文件编译：

| 场景 | 预期贡献数 | 修复前 | 修复后 |
|---|---:|---:|---:|
| 唯一投影点 | 1 | 0 / FAIL | 1 / PASS |
| 最后cell只有一个点 | 4 | 3 / FAIL | 4 / PASS |
| 无效前缀 + 最后cell多个点 | 4 | 3 / FAIL | 4 / PASS |

三个修复后case都验证最后点实际进入下游排他区间遍历，终点等于总索引数，起点与目标cell保持正确。修复前可执行文件返回1，修复后返回0；原始日志 [geometry_bounds_before_test.log](geometry_bounds_before_test.log)、[geometry_bounds_fixed_test.log](geometry_bounds_fixed_test.log)，状态 [geometry_bounds_test.json](geometry_bounds_test.json)。核心库已实际重新编译。

修复前真实首帧的遗漏贡献位于XY=(359,272)，各通道最大0.0494845；它解释一个遗漏，不能据此把其余FP16误差宣称全部修复。旧中间结果保存在 [pool_intermediate_frame_0_before_geometry_fix.json](pool_intermediate_frame_0_before_geometry_fix.json)，修复后使用新capture再次核验。

复现当前kernel测试（项目根目录）：

```bash
/usr/local/cuda-11.8/bin/nvcc -std=c++14 -arch=sm_89 \
  -Iperception_lab/third_party/Lidar_AI_Solution/CUDA-BEVFusion/src \
  perception_lab/tests/bevfusion_geometry_bounds_test.cu \
  -o /tmp/bevfusion_geometry_bounds_test
/tmp/bevfusion_geometry_bounds_test
```

修复前版本可用 `git show HEAD:CUDA-BEVFusion/src/bevfusion/camera-geometry.cu` 导出，编译时用 `-DCAR_GEOMETRY_SOURCE='"该文件绝对路径"'` 并增加原源码`src/bevfusion`头文件目录。当前测试及旧版本测试的实际编译日志均已保留。
