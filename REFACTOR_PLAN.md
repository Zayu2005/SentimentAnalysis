# SentimentSpider 开源项目重构方案

## 一、问题诊断

### 1.1 硬编码问题（严重程度排序）

| 文件 | 行号 | 问题类型 | 严重性 | 硬编码值 |
|------|------|----------|--------|----------|
| `crawler/trigger.py` | 44 | Windows绝对路径 | **严重** | `E:\develop\anaconda3\envs\sentiment\python.exe` |
| `config/settings.py` | 32 | 数据库密码 | **高** | `"1234"` |
| `config/settings.py` | 31 | 数据库用户 | **高** | `"root"` |
| `config/settings.py` | 20 | 相对路径耦合 | **高** | `MediaCrawler/.env` 固定路径 |
| `analyzer/llm_client.py` | 31,69 | API地址 | 中 | DeepSeek/Qwen URL |
| `fetcher/client.py` | 18 | API地址 | 中 | `https://orz.ai/api/v1` |
| `fetcher/client.py` | 42 | User-Agent | 低 | Windows特定字符串 |
| `crawler/trigger.py` | 80 | 超时时间 | 低 | 600秒 |

### 1.2 安全问题

- [ ] 数据库密码硬编码在代码中
- [ ] `.env` 文件没有加入 `.gitignore`
- [ ] SQL注入风险（字符串拼接查询）
- [ ] 敏感信息打印到控制台

### 1.3 文档缺失

- [ ] 根目录 README.md 几乎为空
- [ ] 无 CONTRIBUTING.md
- [ ] 无 CHANGELOG.md
- [ ] 无 LICENSE 文件
- [ ] 无安装指南
- [ ] 无配置参考文档

---

## 二、重构方案

### Phase 1: 配置系统重构（优先级：最高）

#### 1.1 创建统一的环境变量配置

**新建 `.env.example`**：
```env
# ===================================
# SentimentSpider Configuration
# ===================================

# Application Environment
APP_ENV=development  # development | production

# ===================================
# Database Configuration
# ===================================
MYSQL_DB_HOST=localhost
MYSQL_DB_PORT=3306
MYSQL_DB_USER=root
MYSQL_DB_PWD=your_password_here
MYSQL_DB_NAME=sentiment

# ===================================
# LLM Configuration (选择一个)
# ===================================
# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Qwen (阿里云)
QWEN_API_KEY=your_qwen_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# OpenAI Compatible
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# LLM Settings
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
LLM_DEFAULT_PROVIDER=deepseek

# ===================================
# Hot News Fetcher
# ===================================
HOT_NEWS_API_URL=https://orz.ai/api/v1
HOT_NEWS_TIMEOUT=30

# ===================================
# Crawler Configuration
# ===================================
# Python路径 (留空则自动检测)
CRAWLER_PYTHON_PATH=
# MediaCrawler目录 (留空则自动检测)
MEDIACRAWLER_DIR=
# 爬虫超时(秒)
CRAWLER_TIMEOUT=600
# 最大评论数
CRAWLER_MAX_COMMENTS=10

# ===================================
# Logging
# ===================================
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FILE_MAX_SIZE=10485760
LOG_FILE_BACKUP_COUNT=5
```

#### 1.2 重构 settings.py

```python
# hot_news/config/settings.py
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import os

class AppSettings(BaseSettings):
    """应用配置 - 从环境变量加载"""

    # Environment
    app_env: str = Field(default="development", env="APP_ENV")

    # Database
    db_host: str = Field(default="localhost", env="MYSQL_DB_HOST")
    db_port: int = Field(default=3306, env="MYSQL_DB_PORT")
    db_user: str = Field(default="root", env="MYSQL_DB_USER")
    db_password: str = Field(default="", env="MYSQL_DB_PWD")  # 无默认密码
    db_name: str = Field(default="sentiment", env="MYSQL_DB_NAME")

    # LLM
    deepseek_api_key: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", env="DEEPSEEK_BASE_URL")
    qwen_api_key: Optional[str] = Field(default=None, env="QWEN_API_KEY")
    qwen_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", env="QWEN_BASE_URL")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    llm_default_provider: str = Field(default="deepseek", env="LLM_DEFAULT_PROVIDER")

    # Hot News
    hot_news_api_url: str = Field(default="https://orz.ai/api/v1", env="HOT_NEWS_API_URL")
    hot_news_timeout: int = Field(default=30, env="HOT_NEWS_TIMEOUT")

    # Crawler
    crawler_python_path: Optional[str] = Field(default=None, env="CRAWLER_PYTHON_PATH")
    mediacrawler_dir: Optional[str] = Field(default=None, env="MEDIACRAWLER_DIR")
    crawler_timeout: int = Field(default=600, env="CRAWLER_TIMEOUT")
    crawler_max_comments: int = Field(default=10, env="CRAWLER_MAX_COMMENTS")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = Field(default="logs", env="LOG_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("db_password", pre=True, always=True)
    def validate_db_password(cls, v, values):
        if not v and values.get("app_env") == "production":
            raise ValueError("Database password is required in production")
        return v

    def get_python_executable(self) -> str:
        """智能获取Python路径"""
        if self.crawler_python_path and Path(self.crawler_python_path).exists():
            return self.crawler_python_path

        # 尝试检测conda环境
        import sys
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            if os.name == "nt":  # Windows
                return str(Path(conda_prefix) / "python.exe")
            else:  # Linux/Mac
                return str(Path(conda_prefix) / "bin" / "python")

        return sys.executable

    def get_mediacrawler_dir(self) -> Path:
        """智能获取MediaCrawler目录"""
        if self.mediacrawler_dir:
            return Path(self.mediacrawler_dir)

        # 自动检测：相对于本文件
        return Path(__file__).parent.parent.parent / "MediaCrawler"


# 全局单例
_settings: Optional[AppSettings] = None

def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings
```

#### 1.3 重构 trigger.py

```python
# hot_news/crawler/trigger.py (关键部分)

def trigger_crawl(self, keyword: str, platform: str, max_comments: int = None) -> bool:
    settings = get_settings()

    # 使用配置而非硬编码
    mediacrawler_dir = settings.get_mediacrawler_dir()
    python_exe = settings.get_python_executable()
    max_comments = max_comments or settings.crawler_max_comments
    timeout = settings.crawler_timeout

    cmd = [
        python_exe,
        "main.py",
        "--platform", platform,
        "--lt", "cookie",
        "--type", "search",
        "--keywords", keyword,
        "--max_comments_count_singlenotes", str(max_comments),
        "--save_data_option", "db",
    ]

    # ... 其余代码
```

---

### Phase 2: 安全性修复（优先级：高）

#### 2.1 更新 .gitignore

```gitignore
# Environment
.env
.env.local
.env.*.local
!.env.example

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
.eggs/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
logs/
*.log

# Database
*.db
*.sqlite3
*.sql

# OS
.DS_Store
Thumbs.db

# Project specific
MediaCrawler/browser_data/
.claude/
```

#### 2.2 修复SQL注入

```python
# 修改前 (不安全)
sql += f" AND provider = '{provider}'"

# 修改后 (安全)
sql += " AND provider = %s"
params.append(provider)
cursor.execute(sql, params)
```

---

### Phase 3: 文档完善（优先级：高）

#### 3.1 项目结构

```
SentimentSpider/
├── README.md                 # 主文档
├── README_en.md              # 英文文档
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # 贡献指南
├── CHANGELOG.md              # 变更日志
├── .env.example              # 环境变量模板
├── .gitignore
├── pyproject.toml            # 项目元数据
├── requirements.txt
│
├── docs/                     # 文档目录
│   ├── installation.md       # 安装指南
│   ├── configuration.md      # 配置参考
│   ├── architecture.md       # 架构设计
│   ├── api-reference.md      # API参考
│   ├── troubleshooting.md    # 故障排除
│   └── deployment.md         # 部署指南
│
├── hot_news/                 # 核心模块
│   └── ...
│
└── MediaCrawler/             # 爬虫模块
    └── ...
```

#### 3.2 README.md 模板

```markdown
# SentimentSpider

> 舆情监控与情感分析系统 - 自动获取热点新闻、分析领域匹配、提取关键词并爬取相关内容

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 功能特性

- 多平台热点获取（微博、知乎、百度、B站等15+平台）
- 基于LLM的智能领域匹配分析
- 自动关键词提取
- 多平台内容爬取（小红书、抖音、快手等）
- 完整的CLI工具链

## 快速开始

### 安装

\`\`\`bash
git clone https://github.com/your-username/SentimentSpider.git
cd SentimentSpider
pip install -r requirements.txt
\`\`\`

### 配置

\`\`\`bash
cp .env.example .env
# 编辑 .env 填入你的配置
\`\`\`

### 初始化数据库

\`\`\`bash
python -m hot_news.cli.main config init-db
\`\`\`

### 运行

\`\`\`bash
# 一键执行完整流程
python -m hot_news.cli.main run

# 或分步执行
python -m hot_news.cli.main fetch weibo zhihu    # 获取热点
python -m hot_news.cli.main analyze              # 分析匹配
python -m hot_news.cli.main extract              # 提取关键词
python -m hot_news.cli.main crawl xhs            # 触发爬虫
\`\`\`

## 文档

- [安装指南](docs/installation.md)
- [配置参考](docs/configuration.md)
- [架构设计](docs/architecture.md)
- [API参考](docs/api-reference.md)
- [故障排除](docs/troubleshooting.md)

## 许可证

[MIT License](LICENSE)
```

---

### Phase 4: 代码质量提升（优先级：中）

#### 4.1 添加类型注解

```python
from typing import List, Optional, Dict, Any

def trigger_crawl(
    self,
    keyword: str,
    platform: str,
    max_comments: Optional[int] = None
) -> bool:
    ...
```

#### 4.2 添加单元测试框架

```
tests/
├── __init__.py
├── conftest.py           # pytest fixtures
├── test_config.py        # 配置测试
├── test_fetcher.py       # 获取器测试
├── test_analyzer.py      # 分析器测试
└── test_crawler.py       # 爬虫触发器测试
```

#### 4.3 添加 pyproject.toml

```toml
[project]
name = "sentiment-spider"
version = "1.0.0"
description = "舆情监控与情感分析系统"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your@email.com"}
]
keywords = ["sentiment", "analysis", "crawler", "hot-news"]

dependencies = [
    "typer>=0.9.0",
    "httpx>=0.25.0",
    "pymysql>=1.1.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
hot-news = "hot_news.cli.main:app"
```

---

## 三、实施优先级

### 第一批（立即实施）
1. [x] 修复 trigger.py 硬编码路径
2. [ ] 创建 `.env.example`
3. [ ] 更新 `.gitignore`
4. [ ] 重构 `settings.py` 配置加载

### 第二批（本周内）
5. [ ] 编写根目录 `README.md`
6. [ ] 添加 `LICENSE` 文件
7. [ ] 修复 SQL 注入问题
8. [ ] 创建 `CONTRIBUTING.md`

### 第三批（下周）
9. [ ] 创建 `docs/` 文档目录
10. [ ] 添加 `pyproject.toml`
11. [ ] 编写安装指南
12. [ ] 编写配置参考文档

### 第四批（后续）
13. [ ] 添加单元测试
14. [ ] 添加 CI/CD 配置
15. [ ] 添加 Docker 支持
16. [ ] 发布到 PyPI

---

## 四、预期效果

重构后，其他用户只需：

```bash
# 1. 克隆项目
git clone https://github.com/xxx/SentimentSpider.git
cd SentimentSpider

# 2. 安装依赖
pip install -r requirements.txt

# 3. 复制并编辑配置
cp .env.example .env
vim .env  # 填入自己的数据库和API密钥

# 4. 初始化数据库
python -m hot_news.cli.main config init-db

# 5. 运行
python -m hot_news.cli.main run
```

无需修改任何代码即可使用！
