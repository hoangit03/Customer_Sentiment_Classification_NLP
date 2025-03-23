from fastapi import APIRouter
from app.services.ml_model import predict_sentiment, tokenizer

router = APIRouter()


from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

@router.post("/predict/")
async def predict(input: TextInput):
    sentiment = predict_sentiment(input.text,tokenizer)
    return {"text": input.text, "sentiment": sentiment}

# @router.post("/predict/")
# async def predict(text: str):
#     return {"test": text}