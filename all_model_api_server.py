import os
import json
import logging
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Security, Depends, Request, Form
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from openai import OpenAI
from zhipuai import ZhipuAI
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

curdir = os.path.dirname(os.path.abspath(__file__))
# 创建模板引擎
templates = Jinja2Templates(directory=os.path.join(curdir, 'templates'))

API_KEY = ""
API_KEY_HEADER = APIKeyHeader(name="Authorization")

def get_api_key(api_key_header: str = Security(API_KEY_HEADER)) -> str:
    if api_key_header == f"Bearer {API_KEY}":
        return api_key_header
    raise HTTPException(
        status_code=401,
        detail="Invalid API Key"
    )

class AIClientManager:
    """AI客户端管理器，负责初始化和管理所有AI服务提��商的客户端"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.clients: Dict = {}
            self.all_models = {}
            self.configured = False
    
    def configure_clients(self, config_data: dict):
        """根据配置数据初始化客户端"""
        try:
            # 提取并设置API key
            global API_KEY
            API_KEY = config_data.pop("myapikey", "")
            if not API_KEY:
                raise ValueError("必须提供myapikey")
            
            # 清除现有客户端
            self.clients.clear()
            self.all_models.clear()
            
            # 智谱AI是必需的
            if "zhipu" not in config_data or not config_data["zhipu"].get("key"):
                raise ValueError("智谱AI配置是必需的")
            
            # 配置所有模型
            for provider, config in config_data.items():
                if not config.get("key"):  # 跳过没有key的配置
                    continue
                
                # logger.info(f"Processing {provider} config: {json.dumps(config, indent=2)}")
                
                # 只保留必要的字段
                clean_config = {
                    "url": config.get("url", ""),
                    "provider": provider,
                    "key": config["key"]
                }
                
                self.all_models[provider] = clean_config
                
                try:
                    if provider == "zhipu":
                        self.clients[provider] = ZhipuAI(api_key=config["key"])
                    else:
                        try:
                            # 直接使用参数初始化，避免使用字典解包
                            self.clients[provider] = OpenAI(
                                api_key=config["key"],
                                base_url=config.get("url") if config.get("url") else None
                            )
                            logger.info(f"Successfully initialized {provider} client")
                        except TypeError as te:
                            logger.error(f"TypeError initializing {provider} client: {str(te)}")
                            # 尝试不带base_url初始化
                            if "base_url" in str(te):
                                self.clients[provider] = OpenAI(api_key=config["key"])
                                logger.info(f"Successfully initialized {provider} client without base_url")
                    logger.info(f"Successfully initialized {provider} client")
                except Exception as e:
                    logger.error(f"Failed to initialize {provider} client: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    
            self.configured = True
            return True
                    
        except Exception as e:
            logger.error(f"Error configuring AI clients: {str(e)}")
            raise
    
    def get_client(self, provider: str):
        """获取指定提供商的客户端"""
        if not self.configured:
            raise HTTPException(status_code=503, detail="AI服务尚未配置，请先访问配置页面")
        if provider not in self.clients:
            raise HTTPException(status_code=404, detail=f"Provider {provider} not found or not initialized")
        return self.clients[provider]
    
    def get_model_config(self, provider: str):
        """获取指定提供商的配置"""
        if not self.configured:
            raise HTTPException(status_code=503, detail="AI服务尚未配置，请先访问配置页面")
        if provider not in self.all_models:
            raise HTTPException(status_code=404, detail=f"Provider {provider} not found")
        return self.all_models[provider]

# Initialize FastAPI app
app = FastAPI(title="Unified AI API Server")

# Initialize AI client manager
client_manager = AIClientManager()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(default=0.7)
    stream: bool = Field(default=False)
    max_tokens: Optional[int] = Field(default=None)

def get_provider_and_model(model: str):
    """Extract provider and actual model from the model string"""
    try:
        parts = model.split("|")
        if len(parts) < 3:
            raise HTTPException(status_code=404, detail="Invalid model format. Expected format: test|provider|model|备注")
        
        test_flag, provider_name, model_name = parts[:3]
        logger.info(f"test_flag: {test_flag}, provider_name: {provider_name}, model_name: {model_name}")
        
        if test_flag != "test":
            raise HTTPException(status_code=404, detail="Model must start with 'test' prefix")
            
        if provider_name not in client_manager.all_models:
            raise HTTPException(status_code=404, detail=f"Unsupported provider: {provider_name}")
            
        return client_manager.get_client(provider_name), model_name
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid model format. Expected format: test|provider|model")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, request_headers: Request, api_key: str = Depends(get_api_key)):
    try:
        logger.info(f"Received chat completion request for model: {request.model}")
        logger.info(f"Messages: {request.messages}")
        
        provider, actual_model = get_provider_and_model(request.model)
        logger.info(f"Using provider: {provider} with actual model: {actual_model}")

        # Force stream to False for now as we're not handling streaming responses yet
        request.stream = False
        
        # Handle ZhipuAI differently as it has a different API
        if isinstance(provider, ZhipuAI):
            response = provider.chat.completions.create(
                model=actual_model,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=request.temperature,
                stream=request.stream,
            )
        else:
            response = provider.chat.completions.create(
                model=actual_model,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=request.temperature,
                stream=request.stream,
                max_tokens=request.max_tokens if request.max_tokens else None
            )

        # Convert response to a serializable format for non-streaming response
        try:
            return {
                "id": str(response.id),
                "object": "chat.completion",
                "created": int(response.created) if hasattr(response, 'created') else int(time.time()),
                "model": str(response.model),
                "choices": [
                    {
                        "index": choice.index if hasattr(choice, 'index') else 0,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else "stop"
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                    "total_tokens": response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                } if hasattr(response, 'usage') and response.usage else {}
            }
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            # Fallback response format if the standard format fails
            return {
                "id": "fallback_" + str(int(time.time())),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": actual_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {}
            }

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def config_page(request: Request):
    """配置页面"""
    return templates.TemplateResponse(
        "config.html",
        {
            "request": request, 
            "configured": client_manager.configured
        }
    )

@app.post("/configure")
async def configure(config: str = Form(...)):
    """处理配置提交"""
    try:
        config_data = json.loads(config)
        logger.info(f"Received configuration data: {json.dumps(config_data, indent=2)}")
        client_manager.configure_clients(config_data)
        return RedirectResponse(url="/", status_code=303)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)