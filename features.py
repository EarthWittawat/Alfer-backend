import numpy as np
import pandas as pd
import glob
import json
import numpy as np
import base64
import os
import requests
from transformers import pipeline
access_token = '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'
API_URL_airesearch = "https://api-inference.huggingface.co/models/airesearch/wav2vec2-large-xlsr-53-th"
headers = {"Authorization": f"Bearer {access_token}"}
def text_classification(text: str, return_id=True) -> str:
    classifier = pipeline("zero-shot-classification",model="joeddav/xlm-roberta-large-xnli",token=access_token)
    result = classifier(text,candidate_labels=["ความปลอดภัย","การศึกษา", "การเดินทาง", "สิ่งแวดล้อม"])
    return result
def query(API_URL,data):
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
    
    