from fastapi import FastAPI
from app.v1.endpoints import predict

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "message": "This is your item!"}


app.include_router(predict.router, prefix="/api")