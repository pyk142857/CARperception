# 固定GRAE CPU几何语义（待实测剖析补充）

参考为既有 `tools/grae3dmot_2025.py:frame_arrays` 与固定 SimpleTrack依赖，仅调用NMS，不运行旧跟踪器。依赖逐文件实际hash将在cpu_profile.json中保存；当前源码未修改。

- 七类allowlist先执行，原始global xyz/wlh/wxyz/xyvelocity和score以float64读取；非有限几何、非正尺寸、非单位四元数在reader边界拒绝。BBox yaw采用pyquaternion Quaternion.yaw_pitch_roll[0]，模型yaw数组保留原arctan2表达式，二者不能混为同一个优化入口。
- BBox字段w=size[0],l=size[1],h=size[2]。先运行NMS(.1,high=1.0,yaw=.3)，之后才score>=.1；不提前去掉低分框。
- 原np.argsort(scores)[::-1]及整数classes原样保留。同分排序不能擅自改为stable。
- BBoxCoarseFilter gsize=100、scaler=100，四个min/max角点floor网格组合生成键与set；只对当前order中粗筛相关且同类者算IoU。网格键碰撞及原set关系都属于参考行为，不能悄悄改为理想网格或全配对抑制。
- 多边形角点采用原BBox.box2corners2d；第一候选CPU以float64按原方法计算角点并打包，GPU只计算交集/原式高度/IoU与贪心抑制。这样不引入GPU三角函数替代误差，CPU角点打包时间计入预处理层。
- 原高度交叠是 `max(0,min((za+ha/2)-(zb-hb/2),(zb+hb/2)-(za-ha/2)))`，未再min(ha,hb)。不同高度包含时不等于物理标准，可能产生IoU>1，必须保留。
- 体积union+1e-5；严格IoU>threshold_low才删除，不是>=。threshold_high/yaw投票分支会创建avg_bbox，但没有把它写回dets或返回；可见输出仍是原index/type，不能新增平均框副作用。
- 正常正尺寸框自IoU应使其移出order；极小体积导致自IoU<=.1的合成输入可能不终止，须在独立受限进程记录reference_failure，不用无穷循环阻塞全实验。
- typed输入：float64 geometry，int64 order/classes，int32 eligible；整数NMS indices/order/types精确比较，不允许KernelBench统一precision转换破坏类型。FP64误差预算atol1e-8/rtol1e-7，真实IoU接近阈值时可显式CPU复核，比例与耗时必须记录。

目前文档描述源码检查，尚不代表GPU候选已实现或等价通过。

## 本次实际验证补充

FP64 TileLang裁剪采用相对首角点平移，保持原CPU计算的角点、排序和粗筛；`--fmad=false`降低浮点重排差异。GPU抑制是单线程按原order贪心遍历，当前未使用64位bitmask；仍实测63/64/65及最大199、capacity256边界。

120帧全量NMS与模型输入字节精确；9个真实代表帧比较所有eligible pair几何。另有20组合成常规fixture及独立原reader无效输入／上游不终止边界记录。实际120帧阈值CPU复核为0，合成阈值附近复核6对；所有复核和D2H成本都包含在完整NMS计时。

默认high=1的投票分支在正常有效几何中不可达：交集面积≤min(Aa,Ab)，原式交叠高度≤(ha+hb)/2，因此2V交≤Va+Vb，带+1e-5分母后IoU<1。近yaw=.3的高度包含用例已实测，原输入BBox不变。该论证不替代有限精度实际比较，也不隐瞒原始高度公式不是标准物理交叠。

运行入口直接处理不支持的容量／阈值时显式报错，未自动改为近似NMS。具体证据见kernel_correctness.json、kernel_boundary_checks.json及grae_gpu/official_equivalence.json。
