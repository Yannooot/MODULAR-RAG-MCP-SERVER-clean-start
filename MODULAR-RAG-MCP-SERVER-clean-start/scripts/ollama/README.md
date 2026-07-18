# Windows Ollama 与 bge-m3 启动教程

本目录提供 Windows 启动脚本，用于启动本机 Ollama API 服务，并检查
`bge-m3` Embedding 模型是否已经安装。

## 1. 前置检查

在 Anaconda Prompt、CMD 或 PowerShell 中执行：

```cmd
ollama --version
```

如果找不到 `ollama` 命令，请先安装 Ollama，并重新打开终端。

首次使用 `bge-m3` 时执行：

```cmd
ollama pull bge-m3
```

检查模型：

```cmd
ollama list
```

## 2. 启动 Ollama API

如果当前目录是外层仓库：

```text
F:\project\RAG\MODULAR-RAG-MCP-SERVER-clean-start
```

执行：

```cmd
.\MODULAR-RAG-MCP-SERVER-clean-start\scripts\ollama\start_ollama.bat
```

如果当前目录已经是内层项目：

```text
F:\project\RAG\MODULAR-RAG-MCP-SERVER-clean-start\MODULAR-RAG-MCP-SERVER-clean-start
```

执行：

```cmd
.\scripts\ollama\start_ollama.bat
```

脚本会自动切换到项目根目录、激活 `.venv`，然后运行
`start_ollama.py`。服务已经启动时，脚本会直接复用现有服务。

正常输出：

```text
Ollama service is ready at http://localhost:11434.
Embedding model 'bge-m3' is installed.
```

## 3. 测试 Embedding API

以下命令是 PowerShell 语法。如果当前使用 Anaconda Prompt 或 CMD，请先执行：

```cmd
powershell
```

然后将下面整段粘贴到 PowerShell：

```powershell
$body = @{
    model = "bge-m3"
    input = @("你好，这是一次向量测试")
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Uri "http://localhost:11434/api/embed" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$result.embeddings.Count
$result.embeddings[0].Count
```

预期输出：

```text
1
1024
```

`1` 表示生成了一个向量，`1024` 表示 `bge-m3` 的向量维度。

查看向量的前五个数字：

```powershell
$result.embeddings[0][0..4]
```

## 4. 配置项目

确认 `.env` 包含：

```env
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_API_KEY=
```

本地 Ollama 不需要 API Key，因此 `EMBEDDING_API_KEY` 留空。

确认 `config/settings.yaml` 使用 `bge-m3`：

```yaml
embedding:
  provider: ollama
  model: bge-m3
  api_key: ${EMBEDDING_API_KEY}
  base_url: ${EMBEDDING_BASE_URL}
```

## 5. 通过项目代码验证

在内层项目目录执行：

```cmd
.\.venv\Scripts\activate.bat
python -c "from core.settings import load_settings; from libs.embedding.embedding_factory import EmbeddingFactory; e = EmbeddingFactory.create(load_settings()); v = e.embed(['你好，这是项目调用测试']); print(len(v), len(v[0]))"
```

预期输出：

```text
1 1024
```

## 6. 常见问题

### 找不到启动脚本

先确认自己位于外层仓库还是内层项目，再使用第 2 节对应的命令。

### 提示 Ollama 不存在

确认 Ollama 已安装，并关闭当前终端后重新打开：

```cmd
ollama --version
```

### 提示 bge-m3 未安装

执行：

```cmd
ollama pull bge-m3
```

### 端口无法访问

检查 `11434` 端口：

```cmd
netstat -ano | findstr 11434
```

### 切换模型后检索异常

不同 Embedding 模型的向量不能混用。切换到 `bge-m3` 后，需要清理旧向量并重新摄取文档。
