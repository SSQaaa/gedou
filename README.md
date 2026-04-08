# gedou

本仓库是一个实现基于 AprilTag 与 YOLOv5 的实时目标检测与智能决策系统，面向嵌入式端侧应用场景。

针对传统视觉系统在实时性、延迟和资源占用方面的不足，本项目采用轻量化方案，融合：

- **多线程架构**：采集、检测、决策/通信解耦，降低阻塞与抖动
- **ROI 裁剪**：减少不必要的推理区域，提升帧率并降低端侧算力压力
- **优先级通信机制**：用于串口下发控制指令时的“优先级 + 超时回退”策略
- **硬件/带宽优化**：在端侧资源约束下尽量提高吞吐与稳定性

> 说明：本文档中的 `systemd` 自启动与日志查看属于**工具方法**说明，不代表项目全部功能。

---

## 1. 运行环境

- 推荐使用 conda 环境：`gedou`
- 设备端通过 `systemd` 配置开机自启动，确保上电即运行 `main.py`

---

## 2. 创建 systemd 服务（开机自启动）

创建文件：

```bash
sudo nano /etc/systemd/system/winnnnn.service
```

写入内容：

```ini
[Unit]
Description=gedou autostart (run main.py in conda env)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=radxa
WorkingDirectory=/home/radxa/fianlcode
Restart=always
RestartSec=1

# 直接调用 conda 环境里的 python，避免依赖 interactive shell
ExecStart=/home/radxa/miniforge3/envs/gedou/bin/python /home/radxa/fianlcode/main.py

# 日志进 journalctl
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

> 说明：这里使用“conda 环境内的 python 绝对路径”启动程序，比 `conda activate` 更稳定。

---

## 3. 生效/启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable winnnnn.service
sudo systemctl restart winnnnn.service
```

查看状态：

```bash
sudo systemctl status winnnnn.service --no-pager -l
```

---

## 4. 查看服务日志

实时跟踪（`Ctrl+C` 退出）：

```bash
sudo journalctl -u winnnnn.service -f
```

---

## 5. 查看 GPIO 归属

```bash
gpioinfo
```
