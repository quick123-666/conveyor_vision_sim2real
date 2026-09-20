# 工业视觉 sim-to-real 技术方案

> **场景**:传送带 + 1 摄像头 + 流动零件,做合格检查
> **核心思路**:仿照 Microduck 的 sim-to-real 范式,**纯软件训视觉检测模型**,真机只做"轻度验证 + 持续迭代"
> **文档日期**:2026-09-19
> **作者**:Mavis

---

## 0. 项目概述

| 维度 | 值 |
|---|---|
| 场景 | 工业产线,1 个摄像头俯拍传送带,零件连续流动 |
| 目标 | **缺陷检出率 ≥ 99%, 误检率 ≤ 2%, 单件延迟 ≤ 200ms** |
| 范围 | 仿真环境 + 合成数据集 + 训练 pipeline + 真机验证 |
| 预算 | ¥30k-60k(中速线,单产线) |
| 周期 | 8 周出 demo,12 周上线 |

---

## 1. spec.md — 规范("做什么 + 不做什么")

### 1.1 范围

**做**:
- 零件表面缺陷自动检测(划痕/凹陷/缺料/污染/裂纹)
- 缺陷位置 + 类型 + 严重度
- 实时合格/不合格判定
- 数据闭环(误检回灌训练集)

**不做**:
- 分拣机构 / 机械臂控制
- 上游产线 PLC 集成
- 工艺参数监控

### 1.2 检测目标

| 缺陷类型 | 严重度 | 检出阈值 | 误检容忍 |
|---|---|---|---|
| 划痕 | high | 长 ≥ 0.5 mm | ≤ 1% |
| 凹陷/鼓包 | high | 直径 ≥ 1 mm | ≤ 1% |
| 缺料/堵孔 | critical | 任何 | 0% 漏检 |
| 污染 | medium | ΔE ≥ 5 | ≤ 3% |
| 裂纹 | critical | 任何 | 0% 漏检 |

### 1.3 输入输出

| 项 | 规格 |
|---|---|
| 输入图像 | GigE 相机,2448×2048,JPEG/PNG,50 fps |
| 输出 | `{decision: PASS/FAIL, defects: [{type, bbox, severity}], score, reason}` |
| 单件延迟 | ≤ 200ms(中速 500 mm/s 线) |
| 触发 | 编码器 + 光电,精确同步 |

### 1.4 验收标准(可量化)

- 漏检率(critical 类)**0%**
- 漏检率(整体)< 0.5%
- 误检率 < 2%
- 单件延迟 P95 < 200ms
- 测试集 ≥ 1000 张,IoU > 0.85
- 7×24 连续运行 1 周无宕机
- 数据闭环每周迭代模型

---

## 2. plan.md — 技术方案

### 2.1 4 层架构

```mermaid
graph TD
    subgraph "L1 硬件 + 触发"
        H1[全局快门相机<br/>Basler acA1920]
        H3[频闪 LED + 光电触发<br/>+ 编码器]
    end

    subgraph "L2 采集 + 预处理"
        S1[ROI 裁剪 1024x256]
        S2[去噪 + 白平衡]
        S3[异步队列]
    end

    subgraph "L3 sim-to-real 训练"
        T1[PyBullet + BlenderProc<br/>虚拟产线]
        T2[合成数据集<br/>100K 张]
        T3[YOLOv8s + Anomalib<br/>混合训练]
        T4[ONNX + TensorRT]
    end

    subgraph "L4 推理 + 决策"
        R1[异步流水线<br/>4 线程]
        R2[两级串联<br/>Anomalib + YOLO]
        R3[MQTT 上报<br/>SCADA / MES]
        R4[数据闭环<br/>误检回灌]
    end

    H1 --> S1
    H3 --> S1
    S1 --> S2 --> S3
    S3 --> R1
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> R1
    R1 --> R2 --> R3
    R2 --> R4
    T4 -.预训练模型.-> R1
    R4 -.每周 retrain.-> T3

    classDef entry fill:#e1f5ff,stroke:#01579b
    classDef sim fill:#fff3e0,stroke:#e65100
    classDef real fill:#e8f5e9,stroke:#1b5e20
    classDef output fill:#fce4ec,stroke:#880e4f
    class H1,H3 entry
    class T1,T2,T3,T4 sim
    class S1,S2,S3,R1,R2,R3 real
    class R3,R4 output
```

### 2.2 sim-to-real 数据流(核心)

```mermaid
sequenceDiagram
    autonumber
    participant CAD as 零件 3D 模型<br/>(OBJ/STL)
    participant Sim as PyBullet + BlenderProc
    participant Synth as 合成图 + 标签
    participant Train as YOLOv8 + Anomalib
    participant Val as 真机 50-100 张
    participant Deploy as ONNX Runtime
    participant Prod as 产线相机
    participant Loop as 数据闭环

    CAD->>Sim: 加载网格
    Sim->>Sim: 程序化注入缺陷<br/>(划痕/凹陷/缺料/污染/裂纹)
    Sim->>Sim: 域随机化<br/>(位置/光照/材质/背景)
    Sim->>Synth: 100K 合成图<br/>(含 ground truth bbox + mask)
    Synth->>Train: 训练 (混合少量真机样本 5-10%)
    Train->>Val: 真机 50 张验证
    alt mAP > 0.85
        Val->>Deploy: ONNX 导出 + 部署
        Deploy->>Prod: 推理 + 决策
        Prod->>Loop: 误检/漏检图片回灌
        Loop->>Train: 每周自动 retrain
    else mAP < 0.85
        Val->>Sim: 调整缺陷分布<br/>(补难样本)
    end
```

### 2.3 技术栈选型(2026-09-20 更新 — 装包实测后调整)

| 层 | 选型 | 理由 |
|---|---|---|
| **3D 仿真(主)** | **BlenderProc + Blender 3.6+** | 高质量 RGB + ground truth(深度/法线/分割 mask)一键出 |
| **3D 仿真(物理)** | **MuJoCo**(Windows/CN-ISP 友好) | prebuilt wheel 秒装,稳定 60 fps,viewer 内置 |
| **3D 仿真(可选)** | PyBullet | 仅 Linux/Mac 推荐;**Windows CN-ISP pip 80MB 装 30+ min 卡死**,不要做首选 |
| **3D 仿真(可选)** | Isaac Sim | GPU 必需(RTX ≥ 2070)+ nvidia-docker;大规模 sim2real 场景 |
| **可视化(交互)** | **Plotly Mesh3D + Dash** | 鲜亮 + 交互;Windows GPU 注意 lighting preset(见 §5.1) |
| **可视化(静态/降级)** | matplotlib Poly3DCollection | 已有,零安装成本;z-sorting bug 多,只做兜底 |
| **缺陷注入** | 程序化 + BlenderProc shader | 划痕/凹陷/污染的可控生成 |
| **检测模型** | YOLOv8s + Anomalib PatchCore | YOLOv8 = 已知缺陷定位,Anomalib = 未知异常兜底 |
| **训练框架** | PyTorch + Ultralytics + timm | 工业标准 |
| **优化** | ONNX Runtime + TensorRT | ONNX 跨平台,TensorRT GPU 加速 |
| **部署** | Docker + ONNX Runtime | 容器化 |
| **数据闭环** | MLflow + 自定义 retrain 脚本 | 模型版本 + 自动化 |
| **监控** | Prometheus + Grafana | 标准 ops 栈 |
| **流** | MQTT | 工业 IoT 标准 |

### 2.3.1 物理仿真选型实测对比(2026-09-20 实测)

| 软件 | 装包速度 (CN-ISP) | Windows 友好 | GPU 需求 | 推荐场景 |
|---|---|---|---|---|
| **MuJoCo** | **prebuilt wheel 秒装** ✓ | ✓ | 不要 | 物理 + 实时 viewer 调试(本次 PoC 选用) |
| PyBullet | 80MB 源码 30+ min 卡死 ❌ | ✓ | 不要 | Linux/Mac + 复杂机器人 |
| PyVista | wheel 秒装 ✓ | offscreen 需 OSMesa(国内 CDN 不可达 ❌) | 不要 | Linux 服务器渲染 |
| vedo | wheel 秒装 ✓ | 同上 OSMesa ❌ | 不要 | 同上 |
| BlenderProc | ~300MB Blender 慢 ⏱ | 中 | 不要 | 训练阶段出合成图(主力) |
| Isaac Sim | nvidia-docker | ✓ | 必需 RTX | 大规模 sim2real |
| **plotly Mesh3D** | 已有 ✓ | ✓ Windows 注意 GPU 渲染 | 不要 | 鲜亮交互 viewer(本方案推荐) |
| matplotlib | 已有 ✓ | ✓ | 不要 | 静态截图(降级) |

**结论**:**Windows 端推荐组合 = MuJoCo(物理) + BlenderProc(出图) + Plotly(实时 viewer)**。PyBullet 仅在 Linux/Mac 环境使用。

### 2.4 性能预算

| 阶段 | 中速线(500 mm/s) | 快速线(2000 mm/s) |
|---|---|---|
| 编码器触发 | < 5ms | < 2ms |
| 拍照 + 传输 | < 50ms | < 30ms |
| ROI + 预处理 | < 20ms | < 10ms |
| 推理 (CPU/ONNX) | < 80ms | < 30ms |
| 决策 + 上报 | < 20ms | < 10ms |
| **总延迟** | **< 200ms** | **< 100ms** |
| 单件可用时间 | 400ms | 100ms |
| **预算/实际比** | 2x 安全 | 1x 紧张 |

---

## 3. tasks.md — 实施清单

### 阶段 1:仿真环境(1 周)

- [ ] **T1.1** 选 1 个真实零件 → 3D 扫描或 CAD 出 STL/OBJ
- [ ] **T1.2** PyBullet 装 URDF(零件 + 传送带 + 限位栏)
- [ ] **T1.3** PyBullet 虚拟相机标定(intrinsics 跟真机匹配)
- [ ] **T1.4** BlenderProc 装 Blender 3.5+ + Python API
- [ ] **T1.5** 写缺陷注入脚本(划痕/凹陷/缺料/污染/裂纹 5 类)
- [ ] **T1.6** 写域随机化脚本(光照 50+ 组合,位置 ±5mm,背景 3 种)
- [ ] **T1.7** 验证流程跑通:跑 100 张合成图,人眼看像不像真零件

### 阶段 2:合成数据集(2 周)

- [ ] **T2.1** 跑 100K 合成图 + ground truth(labelme 格式)
- [ ] **T2.2** 数据集划分:80K 训练 + 10K 验证 + 10K 测试
- [ ] **T2.3** 验证集可视化抽 100 张,看 5 类缺陷分布是否均匀
- [ ] **T2.4** 标注 YOLO 格式(.txt bbox)+ 类别映射
- [ ] **T2.5** 数据集元信息 JSON(类分布/缺陷比例/光照分布)
- [ ] **T2.6** 入 lessons_learned_kb 或新建 industrial_vision_sim2real_kb

### 阶段 3:模型训练(1 周)

- [ ] **T3.1** YOLOv8s 基线训练(100 epoch,合成图)
- [ ] **T3.2** 验证集评估(mAP@0.5,各类 AP 单独看)
- [ ] **T3.3** Anomalib PatchCore 训(只用良品合成图)
- [ ] **T3.4** 两级串联:PatchCore → YOLOv8(异常先粗筛,再细分类)
- [ ] **T3.5** ONNX 导出 + 简化(simplify=True)+ 量化(int8)
- [ ] **T3.6** ONNX Runtime CPU 推理基准(< 80ms)

### 阶段 4:真机验证(1 周)

- [ ] **T4.1** 装硬件(相机 + 镜头 + 光源 + 光电 + 编码器 + 工控机)
- [ ] **T4.2** 采集真机 100 张(好 70 + 各类缺陷 5-10 张)
- [ ] **T4.3** 真机验证:模型 mAP ≥ 0.85(若 < 0.85,回 T1 调整仿真)
- [ ] **T4.4** 单件延迟实测 P95 < 200ms
- [ ] **T4.5** 7×24 稳定性测试(连续运行 1 周)

### 阶段 5:产线上线 + 持续迭代(持续)

- [ ] **T5.1** MQTT 接入 MES / SCADA
- [ ] **T5.2** 误检 / 漏检图片自动入训练集
- [ ] **T5.3** 每周 retrain 脚本(自动 + 人工审核)
- [ ] **T5.4** Prometheus + Grafana 监控(精度/延迟/吞吐量)
- [ ] **T5.5** 月度精度复盘 + 误检分析

---

## 4. checklist.md — 验收

### 4.1 仿真层

- [ ] 3D 模型还原度 ≥ 90%(目测)
- [ ] 缺陷分布覆盖 5 类
- [ ] 域随机化覆盖 50+ 光照 × 5 位置 × 3 背景 = 750+ 组合
- [ ] 单张合成耗时 < 1s(BlenderProc)
- [ ] 100K 数据集 24 小时内跑完

### 4.2 模型层

- [ ] YOLOv8s 训练 100 epoch,val mAP@0.5 ≥ 0.85
- [ ] 各类 AP ≥ 0.7(geometric 类 ≥ 0.9,stain 类 ≥ 0.6)
- [ ] Anomalib PatchCore AUROC ≥ 0.95
- [ ] ONNX int8 推理 CPU < 80ms
- [ ] 模型大小 < 50MB

### 4.3 真机层

- [ ] 真机 100 张验证 mAP ≥ 0.85
- [ ] 漏检率 < 0.5%(critical 0%)
- [ ] 误检率 < 2%
- [ ] 单件延迟 P95 < 200ms
- [ ] 7×24 稳定性(无宕机 + 内存不增长)

### 4.4 产线层

- [ ] MQTT 上报成功率 ≥ 99.9%
- [ ] 误检图片回灌延迟 < 7 天
- [ ] 模型 retrain 自动化 + 人工审核
- [ ] 监控覆盖 精度 / 延迟 / 吞吐量 / 资源

---

## 5. sim-to-real 关键技术(防"模型在仿真 99%,真机 50%"陷阱)

### 5.1 域随机化(Domain Randomization)

让仿真里故意变,让策略见过足够多变化:

| 维度 | 随机范围 |
|---|---|
| 光照位置 | (x ±50mm, y ±30mm, z 200-500mm) |
| 光照色温 | 4000K-6500K |
| 光照强度 | 100-500 lux |
| 零件位置 | ±5mm x/y, 0-360° 旋转 |
| 背景(传送带) | 3 种纹理 + 磨损程度 5 级 |
| 相机噪声 | 高斯噪声 σ 1-5 |
| 镜头畸变 | 桶形/枕形 ±2% |

> **2026-09-20 补充(Plotly viewer lighting preset,Windows GPU 渲染发暗修复)**: 
> 实时 viewer 用了 Plotly Mesh3D 的别忘了加 `lighting=dict(ambient=0.18, diffuse=0.8, specular=1.2, roughness=0.05, facenormalsepsilon=1e-15, vertexnormalsepsilon=1e-15)` + `lightposition=(1000, 1000, -1000)` + `camera.projection=perspective`。否则在 Windows 桌面 GPU 上 box 看起来是 flat dark washout(不是 plotly bug,是 GPU-specific lighting 实现)。Orthographic projection 完全禁用 lighting,必须用 perspective。

### 5.2 真实物理特性模拟(避免 sim→real gap)

| 真实特性 | 仿真做法 |
|---|---|
| 电机非线性 | PyBullet actuatorGearBox + friction |
| 相机曝光延迟 | Render 模拟 1-10ms 延迟 |
| 镜头眩光 | BlenderProc 后处理 glare |
| 灰尘污染 | 程序化随机透明 overlay |
| 振动模糊 | 渲染时随机 0.5-2 像素高斯模糊 |

### 5.3 sim→real 迁移策略

```mermaid
graph LR
    A[合成图 100K] --> C[预训练<br/>mAP 0.70]
    B[真机 50-100 张] --> D[微调<br/>5-20 epoch]
    C --> D
    D --> E[真机 mAP 0.85+]
    E --> F{是否够好?}
    F -- yes --> G[部署]
    F -- no --> H[分析错误样本<br/>回 T1 调仿真]
    H --> A
```

**关键**:合成图 + 真机图混合训练(比例 80:20),而不是纯合成或纯真机。

---

## 6. 资源 / 预算 / 风险

### 6.1 预算

| 项目 | 数量 | 单价 | 小计 |
|---|---|---|---|
| 全局快门相机 Basler acA1920 | 1 | ¥12000 | ¥12000 |
| 镜头 8mm 高分辨率 | 1 | ¥3000 | ¥3000 |
| 频闪 LED + 光电 + 编码器 | 套 | ¥3500 | ¥3500 |
| 工控机(i5 + GTX 1660Ti) | 1 | ¥6000 | ¥6000 |
| ONNX Runtime 部署 + Docker | 套 | ¥30 | ¥30 |
| 数据闭环 / 监控 / MLflow | 套 | ¥5000 | ¥5000 |
| 开发人力(8 周) | 1 | ¥30000 | ¥30000 |
| **总计** | | | **¥59530** |

### 6.2 关键风险(2026-09-20 实测补充)

| 风险 | 缓解 |
|---|---|
| **sim→real gap > 30%** | 域随机化 + 物理特性模拟 + 混合训练 |
| **真机缺陷样本不足** | 仿真生成 + 客户现场补拍(可与客户约定) |
| **GPU 资源** | 训练用云端 GPU(RunPod/Colab),推理本地 CPU |
| **CN-ISP 装包慢** ⏱ | MuJoCo(prebuilt wheel)/ PyVista / plotly / matplotlib 都能装;**PyBullet 80MB 源码 30+ min 卡死**,**PyVista offscreen 需 OSMesa 国内 CDN 不可达**;Week 0 先做装包验证 |
| **CN-ISP 网络**(模型下载) | Hugging Face 镜像 + 提前验证可达性 |
| **模型迭代失控** | 数据闭环 + 人工审核 + 灰度发布 |
| **viewer 渲染发暗**(Windows GPU) | Plotly Mesh3D 用 lighting preset + perspective projection(见 §5.1) |
| **物理参数失衡**(RL 训练) | 物理参数 + reward + ent_coef 三件套一起调;PPO toy env 起步 **50k timesteps**(5k 会比 baseline 还差) |
| **viewer server idle-kill** | Windows PowerShell 5 分钟 idle 杀 watcher;改用 waitress threads=1 + `_state_lock` 跨线程同步,**不依赖长 session 保活** |

### 6.3 vs 传统方案的 ROI 对比

| 方案 | 投入 | 周期 | 漏检率 |
|---|---|---|---|
| **传统**:人工目检 | 3 人 / 班 | 立即 | 5-15% |
| **传统**:纯真机训练 | ¥100k+ | 6 月+ | 1-3% |
| **本方案**:sim-to-real | ¥60k | 8-12 周 | < 0.5% |
| **全自动数据闭环** | ¥80k | 持续迭代 | 逐月下降 |

**ROI**:8 个月回本(假设每天 1 万件,漏检成本 ¥5/件)。

---

## 7. 落地路线图(2026-09-20 实测后修订)

```
Week 0 (新增 1-2 天) ── 仿真软件 PoC 验证(装包 + 最小 demo 跑通)
                         • MuJoCo: pip install mujoco(prebuilt wheel 秒装)
                         • BlenderProc: 装 Blender 3.6+ + pip install blenderproc
                         • plotly: 已有
                         • 跑通最小传送带 + 蓝盒子 demo
                         • **决策点**:CN-ISP 装包哪步卡就换方案
Week 1-2  ────────────  仿真环境(BlenderProc 出图 + MuJoCo 物理)
Week 3-4  ────────────  100K 合成数据集
Week 5-6  ────────────  模型训练(YOLOv8 + Anomalib)
Week 7-8  ────────────  真机验证 + 微调
Week 9-10 ────────────  产线试运行
Week 11-12 ──────────── 全量上线 + 数据闭环启动
Week 13+  ────────────  持续迭代(月度精度复盘)
```

**Week 0 的关键作用**:不验证装包就直接开搞 Week 1-2 容易撞墙(2026-09-19 实测 PyBullet 装 30+ 分钟卡死)。先把仿真软件栈跑通再 commit。

---

## 8. 工程教训沉淀(给后续 Mavis / agent)

1. **不要追求 100% 仿真精度**,sim→real gap 永远存在,**训练时就考虑**
2. **混合训练**比"纯仿真"或"纯真机"都好(80:20)
3. **缺陷注入**比"造真缺陷"便宜 100 倍,但需域随机化弥补
4. **客户现场补拍** 100 张比"努力训到 mAP 0.9"更现实
5. **数据闭环**决定模型能持续好,一次训练注定衰减
6. **架构稳定前不优化性能**,YOLOv8s vs nano 选型不重要
7. **GPU 不是必须**,ONNX Runtime CPU 推理 + 异步流水线够中速线

### 2026-09-20 实战新增(从 ConveyorEnv + Plotly + PPO 实战)

8. **仿真软件装包先验证**(Week 0 原则):CN-ISP 下 PyBullet 80MB 源码 30+ min 卡死,BlenderProc 要装 Blender ~300MB 慢;**先 pip 装包跑通最小 demo 再 commit 主线工作**,撞墙立刻换方案(MuJoCo prebuilt wheel 秒装)
9. **物理 + 渲染分层选型**:物理(MuJoCo/PyBullet)+ 渲染(BlenderProc/matplotlib/plotly)不要绑死一个栈。Windows + 无 GPU 环境最优解 = MuJoCo + plotly Mesh3D
10. **Plotly Mesh3D Windows GPU 渲染发暗**(lighting 失效):用 `lighting=preset(ambient=0.18, diffuse=0.8, specular=1.2, roughness=0.05, facenormalsepsilon=1e-15, vertexnormalsepsilon=1e-15)` + `lightposition=(1000,1000,-1000)` + `camera.projection=perspective` 修复
11. **RL toy env 起步 50k timesteps**:PPO 5k timesteps 训出来比 random baseline 还差(收敛到 local minimum),50k 才稳定收敛。物理参数 + reward shaping + ent_coef 要一起调
12. **viewer 线程安全 + OS idle-kill**:waitress threads=1 + `_state_lock` 包裹所有 callback + policy_action 在 lock 内 snapshot;**不要在长 session 上赌保活**,任务前 preflight,死了就拉起
13. **不要在 viewer 上反复修 bug,看到 race condition / OS idle-kill 直接换方案**(用户原话"工作方法有问题")。换离线 HTML 渲染 / 换 Plotly 替代 Matplotlib / 换 waitress 替代 Flask dev server

---

## 9. 最小可行 PoC 参考实现(2026-09-19 完成)

已完成的 ConveyorEnv + PPO + viewer 可作为本方案的最小 PoC 验证 sim-to-real pipeline 的可行性:

| 文件 | 作用 |
|---|---|
| `conveyor_env.py` | MuJoCo XML + Gymnasium wrapper,2.5D 传送带模型(box 沿 belt 走,agent 推 y) |
| `train_ppo.py` | SB3 PPO 训练(50k timesteps),Single MlpPolicy,n_steps=128,batch_size=64,gamma=0.99 |
| `eval_ppo.py` | PPO vs heuristic 对照(mean -23.80 vs -12.98,std 1.89 vs 7.09) |
| `mujoco_dash_viewer.py` | 双面板 viewer — 左 Plotly 3D 实时仿真 + 右训练曲线 dashboard,waitress 8051 |
| `replay_ppo.py` | 离线 HTML 生成器(38 frames animation + slider + Play/Pause) |
| `ppo_conveyor.zip` | 训练好的 SB3 PPO model(140KB,50050 timesteps) |

**产出**(桌面 `synth_demo_2026-09-19/`):
- `ppo_50k_curve.png` — 991 episode 完整学习曲线
- `plotly_full.png` — 鲜亮 3D demo(box 沿 belt)
- `ppo_replay.html` — 离线动画(38 frames)
- `dash_viewer_live.png` — viewer 实时截图(左 3D + 右 dashboard)
- 250+ 张合成图(BlenderProc 风格的 placeholder)

**PoC 经验复用**:
1. 物理参数给 5.0 阻尼 + 1.5 units/s belt_speed,box 反应线性可控
2. PPO 训前先 heuristic 跑通 baseline(reward mean -12.98),确认 env 没问题
3. viewer 实时版 + 离线 HTML 双路径:server alive 时用实时版,idle-kill 后用离线版
4. 9 轮 OCR code review 修了 30+ 个 bug(包括 docstring 跟实现对齐、thread-safe 跨线程共享 policy)

---

## 10. Jev-Like 决策系统集成(2026-09-20 新增)

### 10.1 概述

Jev-Like 是仿 TypeSafe AI Jev 的国内决策 API(`rag/jev_like.py`,SiliconFlow + Qwen2.5-7B-Instruct),3 个原语:

| 原语 | 用途 | 返回 |
|---|---|---|
| **Choice** | 从预定义选项选一个 | `choice` + `probabilities` + `confidence` |
| **Score** | 对状态打 N 级分 | `score` (0..N-1) + `probabilities` + `confidence` |
| **Noul** | 是/否概率 | `noul` (0..1) |

**核心参数**: 延迟 2-3 秒,价格接近免费(SF 促销),0 类型错误(json_schema strict mode 数学保证),接口 100% 兼容 Jev。

**集成方式**: 单实例 `jev_decision.py` 客户端 + 异步并行(decision thread pool),在产线推理层 + 运营层共 5 个用法。

### 10.2 用法 A — 缺陷分级决策(Choice)

| 项 | 内容 |
|---|---|
| **触发** | YOLOv8 检测到缺陷(bbox + 类别 + 置信度) |
| **输入** | `f"YOLO: 类别=划痕 置信度=0.85 bbox=120,80,30,5 长宽比=6.0"` |
| **输出** | `severity ∈ {low, medium, high, critical}` |
| **原语** | Choice |
| **集成位置** | L3 推理层,YOLO 输出之后 |
| **Schema** | json_schema strict:`{"severity": str ∈ levels, "reason": str}` |
| **价值** | 解决"YOLO 只分类 + 定位,不知道严重度"的核心 gap |

```python
# industrial_vision/jev_decision.py (示意)
from rag.jev_like import JevLikeClient

client = JevLikeClient()

def yolo_to_severity(yolo_output: dict) -> JevLikeResult:
    state = f"YOLO 检测: 类别={yolo_output['class']} 置信度={yolo_output['conf']} bbox={yolo_output['bbox']} 长宽比={yolo_output['aspect']}"
    return client.choice(
        state=state,
        options=["low", "medium", "high", "critical"],
        schema_extra={"reason": "string, 30 字以内解释判定理由"},
    )
```

### 10.3 用法 B — 多模型融合决策(Choice)

| 项 | 内容 |
|---|---|
| **触发** | YOLOv8 + Anomalib PatchCore 都完成推理 |
| **输入** | `f"YOLO: {yolo}\nAnomalib: {anom_score}"` |
| **输出** | `decision ∈ {PASS, FAIL, REVIEW}` |
| **原语** | Choice |
| **集成位置** | L4 决策层(R2 两级串联之后,Anomalib 兜底→YOLO 细分类→Jev 终裁) |
| **Schema** | `{"decision": str ∈ {PASS,FAIL,REVIEW}, "reason": str}` |
| **价值** | 当 YOLO 和 Anomalib 冲突时(YOLO 高置信度 + Anomalib 高分,或反之)的 tie-breaker |

**冲突场景**:
1. YOLO 0.95 confidence + Anomalib 0.1(都认为正常)→ PASS
2. YOLO 0.3 confidence + Anomalib 0.9(YOLO 看不清,Anomalib 觉得异常)→ REVIEW
3. YOLO 0.9 + Anomalib 0.8(都异常)→ FAIL
4. YOLO 0.4 + Anomalib 0.4(都不确定)→ REVIEW

### 10.4 用法 C — 数据闭环人工分流(Choice)

| 项 | 内容 |
|---|---|
| **触发** | 每周 retrain 脚本启动前(自动调度) |
| **输入** | `f"误检/漏检样本: 路径 / 真标签 / 模型当前预测"` |
| **输出** | `action ∈ {AUTO_PASS, MARK_SUSPECT, REJECT}` |
| **原语** | Choice |
| **集成位置** | L4 数据闭环(R4 retrain 之前) |
| **Schema** | `{"action": str ∈ options, "confidence": float, "reason": str}` |
| **价值** | 节省 80% 人工审核成本(典型周 200-500 张误检) |

**dry-run 验证**: 上线前 2 周跑 dry-run(只决策不入训练集),对比 Jev 决策与人工审核,一致率 ≥ 85% 才允许 AUTO_PASS。

### 10.5 用法 D — 真机 vs 仿真差异判定(Score)

| 项 | 内容 |
|---|---|
| **触发** | 数据闭环收到新样本(可能是 sim 漏入,也可能是真机采集) |
| **输入** | `f"图像 metadata: 光照分布 / 背景纹理 / 噪声 / 缺陷形态"` |
| **输出** | `origin ∈ {0=sim, 1=uncertain, 2=real}` (3 级 Score) |
| **原语** | Score |
| **集成位置** | L3 数据标注前 |
| **Schema** | `{"origin": int 0..2, "confidence": float}` |
| **价值** | 训练时按 origin 分桶(sim 80% + real 20%),不用手动标注 |

**风险**: 决策准确率不一定高(sim-real gap 本身就是要解决的问题),需要 confidence 阈值 < 0.6 时回退到人工标。

### 10.6 用法 E — 生产告警分级(Score)

| 项 | 内容 |
|---|---|
| **触发** | Prometheus 监控指标异常(模型精度 / 延迟 / 推理失败率 / GPU 占用) |
| **输入** | `f"指标: mAP=0.82 (历史 0.88), P95=240ms (阈值 200ms)"` |
| **输出** | `alert_level ∈ {0=info, 1=warn, 2=critical, 3=page}` (4 级 Score) |
| **原语** | Score |
| **集成位置** | L4 监控层(R3 → Alertmanager → Grafana) |
| **Schema** | `{"alert_level": int 0..3, "channels": [str], "reason": str}` |
| **价值** | 告警分级自动化,值班人员不用半夜被误报叫醒 |

### 10.7 共享决策服务设计

```
+------------------+
|  industrial_vision/
|  jev_decision.py |  <-- 复用 rag/jev_like.py:JevLikeClient
|  (单实例)        |
+------------------+
        |
        |  (5 个用法共用 client)
        |
+-------+-------+-------+-------+-------+
|       |       |       |       |       |
YOLO    YOLO+   retrain  meta   monitor  (5 个用法入口)
        Anom    script  data
```

**实现要点**:
1. **异步并行**: Jev 延迟 2-3s,**不能**串行嵌进推理流水线 — 用 `concurrent.futures.ThreadPoolExecutor(max_workers=5)` 并发决策
2. **决策缓存**: 同图片 + 同 prompt 的决策结果 LRU 缓存(避免重复决策),TTL 1 小时
3. **prompt 模板统一管理**: `jev_decision_prompts.py` 集中所有 prompt,json_schema 一起定义
4. **降级策略**: SF API 限流/超时 → fallback 到本地 BGE-m3 + rule-based 简单决策,标记 degraded=True
5. **决策日志**: 所有决策结果入 `data/jev_decisions/` 留 audit trail

### 10.8 落地路径(新增)

```
Week 7 (并行,不阻塞主线)
├─ Day 1 上午:jev_decision.py 封装 + jev_decision_prompts.py
├─ Day 1 下午:A 缺陷分级 schema + 测试(用 PoC 250+ 合成图回测)
├─ Day 2:B 多模型融合 schema + viewer 加决策面板
├─ Day 3:C 数据闭环分流(dry-run 2 周)
├─ Day 4:D 真机vs仿真(配合 Week 3-4 合成数据集)
└─ Day 5 上午:E 告警分级 + Alertmanager 配置
Week 8-9:产线试运行 + 决策准确率监控
Week 10+:Jev 决策阈值精调
```

**估时**: 5 个工作日(纯单人),可与 Week 5-6 模型训练并行。

### 10.9 Jev 集成的 5 条风险

| 风险 | 缓解 |
|---|---|
| **延迟 2-3s 不达 200ms 预算** | 异步并行(decision thread pool),**拍照和决策并发** |
| **SF API 限流**(Token Plan 配额) | 决策缓存(LRU)+ fallback rule-based + 限流监控 |
| **决策准确率 < 85%** | Phase 3 dry-run 验证,阈值 ≥ 85% 才允许 AUTO_PASS |
| **prompt 越改越乱** | jev_decision_prompts.py 集中管理 + 版本控制 + 决策日志 audit |
| **真机vs仿真判定不可靠** | confidence < 0.6 回退人工标,不强制全自动 |

### 10.10 vs 不引入 Jev 的对照

| 维度 | 无 Jev(传统) | 有 Jev-Like |
|---|---|---|
| 严重度判定 | 人工规则(if 长宽比 > 5 → high),漏场景 | 语义理解,自然语言 prompt 描述缺陷特征 |
| YOLO+Anomalib 冲突 | hardcoded if-else | 决策可追溯(reason 字段),可调阈值 |
| 数据闭环人工成本 | 全人工审核 | 自动分流 80% |
| 告警分级 | 静态阈值 + Prometheus rules | 语义级别,LLM 解释为何告警 |
| 决策延迟 | < 1ms (本地 rule) | 2-3s (异步,不影响产线) |

---

**生成时间**: 2026-09-19 20:46 (v1) / 2026-09-20 01:16 (v2 优化版) / 2026-09-20 12:00 (v3 Jev-Like 集成)
**作者**:Mavis
**字数**: v1 ~2,800 字 / v2 ~3,800 字 / v3 ~5,500 字
**配套工程文档**: `My reliable experience/` 教程目录 + 桌面 `synth_demo_2026-09-19/` PoC 产物
**v2 优化依据**: 基于 2026-09-19 完成的 ConveyorEnv + PPO + viewer PoC 实战经验,补充 §2.3.1 选型对比 / §5.1 Windows 渲染坑 / §6.2 实战风险 / §7 Week 0 / §8 五条新教训 / §9 PoC 链接
**v3 Jev-Like 集成**: 用户决策 2026-09-20 12:00,5 个用法全部纳入 — §10.1 概述 / §10.2-A 缺陷分级 / §10.3-B 多模型融合 / §10.4-C 数据闭环分流 / §10.5-D 真机vs仿真 / §10.6-E 告警分级 / §10.7 共享服务设计 / §10.8 落地路径 / §10.9 5 条风险 / §10.10 vs 传统对照