# Vllm-Ascend-Serviceparam-Optimizer-Plugins

## 项目简介

此目录下是一个用于 vLLM 双机服务化参数自动寻优的插件，旨在实现基于vllm-ascend的双机服务化参数自动寻优。

## 功能特性

- **双机服务管理**：同时管理本地和远程（通过 SSH）的 vLLM Docker 服务
- **参数自动寻优**：集成到 msserviceprofiler 框架中，支持双机场景下自动优化 vLLM 服务参数

## 项目结构

```
Vllm-Ascend-Serviceparam-Optimizer-Plugins/
├── src/
│   └── vllm_msserviceprofiler/
│       ├── __init__.py       # 插件注册入口
│       ├── benchmark.py      # 基准测试实现
│       ├── settings.py       # 配置管理
│       └── simulator.py      # 双机服务模拟器
├── README.md                 # 项目说明文档
└── pyproject.toml            # 项目依赖配置
```

## 核心组件

### 1. 插件注册 (`__init__.py`)
- 注册自定义 vLLM 模拟器
- 注册自定义设置

### 2. 性能测试 (`benchmark.py`)
支持自定义性能指标，包括生成速度、首 token 时间、输出 token 时间、成功率和吞吐量，代码中并没有使用。**可忽略此文件**。

### 3. 配置管理 (`settings.py`)
- `CusSettings` 类：管理插件配置

### 4. 双机服务模拟器 (`simulator.py`)
- `CustomVllmDockerSimulator` 类：管理本地和远程 vLLM 服务
- 通过SSH完成Docker容器中的vLLM服务的启停

## 安装方法

1. 克隆项目：
```bash
git clone <repository-url>
cd contrib
```

2. 安装依赖：
参考[自定义插件开发指导](https://gitcode.com/Ascend/msmodeling/blob/develop/docs/zh/user_guide/optix_plugin_user_guide.md)完成插件安装
```bash
pip install -e .
```


## 注意事项

1. 确保本地和远程机器已安装 Docker
2. 确保 SSH 连接已配置免密
3. 确保双机 vLLM-Ascend 服务可以正常运行并完成aisbench/vllm bench性能测试
4. 已完成安装msserviceparam_optimizer
