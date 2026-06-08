from fastapi import FastAPI
from pydantic import BaseModel
from app import rag_query

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

app = FastAPI(title="RAG 知识库 API", description="私有化知识库问答接口")

@app.get("/")
def root():
    return {"message": "RAG API 服务已启动", "docs": "/docs"}

@app.post("/rag/query")
def query(req: QueryRequest):
    # rag_query 返回元组，第一个元素是答案
    result = rag_query(req.question, top_k=req.top_k)
    answer = result[0]  # 取第一个返回值作为答案
    return {
        "question": req.question,
        "answer": answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)