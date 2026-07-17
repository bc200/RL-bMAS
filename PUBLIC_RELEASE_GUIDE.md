# 公开仓库使用与安全说明

本文档说明本项目公开发布时的代码范围、环境配置和凭据安全要求。项目主体基于 `verl-agent`，用于多轮智能体环境中的强化学习训练与评测。

## 发布范围

- `OASIS/` 是本地保留目录，已通过根目录 `.gitignore` 强制排除，不属于公开仓库内容。
- `.env`、私钥、证书、凭据文件、运行日志、检查点、轨迹和实验输出均不得提交。
- `.env.example` 只列出变量名和非敏感默认值，可安全纳入版本控制。
- 第三方数据和代码应继续遵守各自许可证；大型数据集建议使用原始下载链接、Git LFS 或独立数据仓库管理。

## 安装

建议使用隔离的 Python 环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

具体训练、环境安装和示例命令见主 [README](README.md)。

## 配置 API 与工具凭据

先复制示例文件，再只在本机填入密钥：

```bash
cp .env.example .env  # Windows PowerShell: Copy-Item .env.example .env
```

代码不再包含 API Key 字面量。可按实际使用的供应商设置以下变量：

| 用途 | 环境变量 |
| --- | --- |
| 通用 OpenAI 兼容客户端/Embedding | `OPENAI_API_KEY`、`CUSTOM_API_KEY`、`CUSTOM_BASE_URL` |
| DMX | `DMX_API_KEY`、`DMX_BASE_URL` |
| DeerAPI | `DEER_API_KEY`、`DEER_BASE_URL` |
| Together AI | `TOGETHER_API_KEY`、`TOGETHER_BASE_URL` |
| DashScope | `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` |
| DeepInfra | `DEEPINFRA_API_KEY`、`DEEPINFRA_BASE_URL` |
| SiliconFlow | `SILICONFLOW_API_KEY`、`SILICONFLOW_BASE_URL` |
| Google Serper 搜索工具 | `SERPER_API_KEY` |
| Wikimedia 请求标识与可选代理 | `WIKIMEDIA_USER_AGENT`、`WIKIMEDIA_PROXY_URL` |
| WebShop 服务 | `WEBSHOP_URL` |
| 本地轨迹和 Weights & Biases 输出 | `VERL_TRAJECTORY_SAVE_DIR`、`WANDB_DIR` |

只配置当前任务实际需要的供应商即可。缺少所需密钥时，相关客户端会给出明确错误，而不会回退到仓库内置凭据。

## 提交前安全检查

1. 确认 `.env` 与 `OASIS/` 未进入暂存区。
2. 运行仓库中的 TruffleHog GitHub Actions 工作流，检查已验证和未知凭据。
3. 搜索常见密钥、个人邮箱、绝对用户目录和私钥头。
4. 若密钥曾经写入源码、日志、终端记录或 Git 历史，应立即在供应商控制台撤销并轮换；仅从文件中删除并不能使旧密钥失效。
5. 提交时显式指定文件范围，不使用未经审查的全量暂存。

## GitHub Actions 凭据

CI/CD 所需凭据应存放在 GitHub 仓库或环境级 Secrets 中，并在工作流中通过 `${{ secrets.NAME }}` 引用。不要把真实值写入 YAML、示例命令、Issue、PR 描述或测试夹具。
