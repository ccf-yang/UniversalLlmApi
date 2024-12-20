# AI Service API Server

统一的AI服务API服务器，支持多个AI提供商的动态配置和管理。
- 只要某个厂商的api支持如下格式的python的api调用，就可以使用该接口去请求，注意配置和调用时的模型名。
```
from openai import OpenAI

client = OpenAI(
    api_key="MODELSCOPE_SDK_TOKEN", 
    base_url="https://api-inference.modelscope.cn/v1"
)
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct", 
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': '用python写一下快排'
        }
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='', flush=True)

```

## 特性

- 支持多个AI服务提供商（智谱AI、Novita AI、PPInfra AI、ModelScope AI）
- 动态配置支持，可通过Web界面实时更新配置
- 安全的API认证机制
- 兼容OpenAI API格式
- Docker容器化部署支持

## 快速开始

### Docker部署

1. 构建Docker镜像：
```bash
docker build -t ai-service-api .
```

2. 运行容器：
```bash
docker run -d -p 7778:7778 ai-service-api
```

### 配置服务

1. 访问配置页面：`http://localhost:7778`

2. 在配置页面输入JSON格式的配置信息：
```json
{
    "myapikey": "your-api-key",  // 设置API访问密钥
    "zhipu": {
        "url": "",
        "provider": "zhipu",
        "key": "your-zhipu-key"
    },
    "novita": {
        "url": "https://api.novita.ai/v3/openai",
        "provider": "novita",
        "key": "your-novita-key"
    }
}
```

注意：
- `myapikey` 字段必填，用于后续API访问认证
- 至少需要配置智谱AI（zhipu）
- 其他提供商配置可选

### API使用

所有API请求需要在header中包含认证信息：
```
Authorization: Bearer your-api-key
```

#### Chat Completion API

```bash
curl -X POST "http://localhost:7778/v1/chat/completions" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test|zhipu|glm-4",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

模型格式说明：
- 格式：`test|provider|model`
- provider: 提供商标识（zhipu/novita/ppinfra/modelscope）
- model: 具体的模型名称

## 安全说明

1. API密钥管理
   - API密钥通过配置JSON的 `myapikey` 字段设置
   - 密钥仅保存在内存中，不会持久化存储
   - 每次重新配置可以更换API密钥

2. 配置安全
   - 配置完成后，敏感信息（如API密钥）会从配置中移除
   - 可以随时通过配置页面更新配置

## 注意事项

1. 配置要求
   - 智谱AI（zhipu）配置是必需的
   - 其他提供商配置可选
   - 必须提供 `myapikey` 用于API认证

2. API访问
   - 所有API请求都需要提供正确的Authorization header
   - Bearer token格式：`Bearer your-api-key`

3. 错误处理
   - 401: API密钥无效
   - 404: 提供商或模型未找到
   - 503: 服务未配置
