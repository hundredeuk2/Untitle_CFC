from fastapi import FastAPI
from pydantic import BaseModel
import inference

class user_input(BaseModel):
    x : str
    
app =FastAPI()

@app.post("/classifier")
def predict(input:user_input):
    result = inference(input.x)
    return result