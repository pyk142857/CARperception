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
