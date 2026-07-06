from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Safety RAG Assistant",
    description="RAG-powered Q&A over industrial safety regulations and incident reports",
)
app.include_router(router, prefix="/api/v1")
