from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from models import analyse, analyse_batch

app = FastAPI(title="Sentiment API", version="1.0.0")

class Inp(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}

@app.post("/predict")
def predict(inp: Inp) -> Dict[str, Any]:
    if inp.text and not inp.texts:
        return {"results": [analyse(inp.text)]}
    if inp.texts:
        return {"results": analyse_batch(inp.texts)}
    return {"results": []}
