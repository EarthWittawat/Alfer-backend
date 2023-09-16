from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import features # all data processing and clustering algorithms will be in this file
import uvicorn
from scipy.io.wavfile import read, write
import io

## allow CORS
origins = ['*']
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def home():
    msg = {
        'message': 'popsloth prototype backend',
        'api_endpoints': ['text_classification'], 
    }
    result = JSONResponse(content=msg)
    return result

@app.get("/api/text_classification")
async def text_classification(text:str):
    mood = features.text_classification(text,return_id = False)
    msg = {
        'message': 'Classify',
        'mood': mood,
    }
    result = JSONResponse(content=msg)
    mood_ = JSONResponse(content=mood['labels'])
    return result
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)