# 恢复工作台深色主题

日期：2026-09-29。

Rerun 从 0.23.4 升级到 0.27.3 后，默认主题改为跟随系统，导致查看器浅色、侧栏深色。此次在查看器启动前设置其持久化 egui 主题为 Dark，并防止旧版本迁移逻辑将其重设为 System。保留已有其他偏好。该兼容处理针对固定的 0.27.3 版本，后续升级应重新验证存储协议。

仅修改 `web/case_browser/app.js` 的主题初始化。完整点云、3D 相机取景和检测数据不变。

验证环境：<http://127.0.0.1:9092/>，Chrome / Playwright，1900×1250 与 700×950。Browser plugin not available，使用已安装的 Playwright。

运行 ` /usr/bin/python3 perception_lab/tests/browser_selection.py --out /tmp/car_dark_theme`，通过事件选择、成员切换、正常框开关、片段回放、FN 选择等检查；无页面错误，主场景只请求一次。桌面截图确认 BEV、3D、文本及时间轴均恢复深色。[交互结果](verification.json)，[恢复后截图](selected_frame2.png)。未验证其他浏览器。
