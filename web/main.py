import sys
import os
import json
import asyncio
import uuid
import time
import base64
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from agent.react_agent import ReactAgent
from model.factory import multimodal_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from utils.logger_handler import logger

app = FastAPI(title="AI Agent Chat")

agent = ReactAgent()
executor = ThreadPoolExecutor(max_workers=4)

# --- Conversation persistence ---
CONVERSATIONS_FILE = os.path.join(os.path.dirname(__file__), "conversations.json")
conversations: dict = {}

def load_conversations():
    global conversations
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                conversations = json.load(f)
        except Exception:
            conversations = {}

def save_conversations():
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

load_conversations()

# --- Static files ---
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# --- Page ---
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# --- Conversation CRUD ---
@app.post("/api/conversations")
async def create_conversation():
    conv_id = str(uuid.uuid4())
    conversations[conv_id] = {
        "id": conv_id,
        "title": "新对话",
        "messages": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    save_conversations()
    return {"id": conv_id}


@app.get("/api/conversations")
async def list_conversations():
    conv_list = []
    for conv in conversations.values():
        # Generate preview from first user message
        preview = ""
        for msg in conv["messages"]:
            if msg["role"] == "user":
                preview = msg["content"][:60]
                break
        conv_list.append({
            "id": conv["id"],
            "title": conv["title"],
            "preview": preview,
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "message_count": len(conv["messages"]),
        })
    conv_list.sort(key=lambda x: x["updated_at"], reverse=True)
    return conv_list


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = conversations.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    conversations.pop(conv_id, None)
    save_conversations()
    return {"ok": True}


# --- Image serving ---
ILLUSTRATION_DIR = None

def _get_illustration_dir():
    global ILLUSTRATION_DIR
    if ILLUSTRATION_DIR is None:
        from utils.path_tool import get_abs_path
        from utils.config_handler import milvus_config
        ILLUSTRATION_DIR = os.path.join(get_abs_path(milvus_config["data_path"]), "插图")
    return ILLUSTRATION_DIR


@app.get("/api/images/{image_name:path}")
async def serve_image(image_name: str):
    """从插图目录加载图片"""
    img_dir = _get_illustration_dir()
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(img_dir, image_name + ext)
        if os.path.exists(path):
            media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            return FileResponse(path, media_type=media_type)
    raise HTTPException(status_code=404, detail="图片不存在")


# --- Chat (SSE streaming) ---
@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()
    conv_id = body.get("conversation_id", "")
    images = body.get("images", [])  # list of base64 image strings

    if not query and not images:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': '请输入问题或上传图片'})}\n\n"]),
            media_type="text/event-stream",
        )

    async def event_stream():
        loop = asyncio.get_event_loop()
        full_response = ""

        try:
            if images:
                # --- 多模态模式：使用视觉模型处理图片 ---
                system_msg = SystemMessage(content="你是一个专业的视觉分析助手。请根据用户上传的图片和问题，给出准确、详细的回答。如果用户没有具体问题，请简要描述图片内容。")

                human_parts = []
                if query:
                    human_parts.append({"type": "text", "text": query})

                for i, img_b64 in enumerate(images):
                    # Ensure base64 has proper data URI prefix
                    if img_b64.startswith("data:"):
                        human_parts.append({"type": "image_url", "image_url": {"url": img_b64}})
                    else:
                        human_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

                messages = [system_msg, HumanMessage(content=human_parts)]

                result = await loop.run_in_executor(executor, multimodal_chat_model.invoke, messages)
                full_response = result.content if hasattr(result, 'content') else str(result)

                # Stream the full response in chunks
                chunk_size = 4
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i + chunk_size]
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                    await asyncio.sleep(0.02)

            else:
                # --- 纯文本模式：使用 Agent（RAG 检索+自行处理） ---
                gen = await loop.run_in_executor(executor, agent.execute_stream, query)

                while True:
                    try:
                        chunk = await loop.run_in_executor(executor, next, gen)
                        if chunk:
                            # 逐字输出，实现打字机效果
                            for char in chunk:
                                full_response += char
                                yield f"data: {json.dumps({'content': char})}\n\n"
                                await asyncio.sleep(0.05)
                    except StopIteration:
                        break

        except Exception as e:
            logger.error(f"[Chat] 处理失败: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Save to conversation history
            if conv_id and conv_id in conversations:
                conv = conversations[conv_id]
                if not conv["messages"]:
                    title = (query[:30] + "…") if len(query) > 30 else query
                    conv["title"] = title or "图片对话"
                conv["messages"].append({
                    "role": "user", "content": query or "[图片]", "timestamp": time.time(),
                    "images": images if images else None
                })
                conv["messages"].append({
                    "role": "assistant", "content": full_response, "timestamp": time.time()
                })
                conv["updated_at"] = time.time()
                save_conversations()

            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
