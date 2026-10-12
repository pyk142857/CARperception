# TensorRT 独立CPU复核

生产与诊断各20次（4 token×5），原生Nx11输出有限，标准化框数量对应；40份文件均记录新哈希。已加载库记录中的文件SHA256重新核验；生产core与隔离diagnostic core路径/哈希不同，pybev和LayerNorm保持原身份，生产14个库/engine/ONNX文件及两次identity一致。

四token的FP32图像、点云及4组标定矩阵与reference预处理输入逐元素完全一致，production/diagnostic也一致。这不等于后续FP16投影或散射逐位相同。

|token|无效索引前缀|有效贡献|参考/TRT贡献数不同的格点|最大数量差|
|---|---:|---:|---:|---:|
|ca9a282c9e77460f8360f564131a8af5|158250|1835478|126|2|
|e0845f5322254dafadbbed75aaa07969|158501|1835227|136|1|
|b26e791522294bec90f86fd72226e35c|159144|1834584|156|1|
|747aa46b9a4641fe90db05d97db2acea|158848|1834880|104|2|

实际interval均正长、连续，末端为排他end且等于完整indices长度；必须跳过无效前缀。双方总贡献数相同，但格点贡献数差异仍存在，不能写geometry逐位一致。

40个module/upstream组合每个均有4份对首跑的比较记录，共5次。记录均有限且逐位相同；独立复核保存的输入/首跑输出哈希。重复raw输出未保存，因此本审阅不能独立重算5次逐位一致，明确区分记录核对与实际数组复算。

8个head replay的marked/unmarked六输出均实际读数组重算，与报告一致；均通过原tensor容差，却全部非逐位相同。因此candidate_source_cases的production_query_attribution=false是正确限制。标记图TopK的[1,324000]分数、[1,200]唯一合法索引、gather值、降序Top200及第200/201选择间隔已独立验证；这些仅属于标记诊断图，不能无条件归给生产head。

优先token747的repeat0/1 sparse-depth节点实际数组统计重新计算，与报告完全一致。未重跑GPU、未修改实现或其他报告。

详情与文件哈希：[independent_review.json](independent_review.json)。
