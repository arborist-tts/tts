from transformers import BarkModel
from pydantic import BaseModel
import torch
from transformers import AutoProcessor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import scipy
import io

# fastapi initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# tts initialization
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark-small")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device) # type: ignore

sampling_rate = model.generation_config.sample_rate # type: ignore

# import scipy
# scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_output[0].cpu().numpy())

class InputData(BaseModel):
    prompt: str
    preset: str

@app.post("/get-speech/")
async def add_song(params: InputData):
    prompt, preset = params.prompt, params.preset

    inputs = processor(prompt, voice_preset=preset)
    speech_output = model.generate(**inputs.to(device))
    
    byte_data = io.BytesIO()
    scipy.io.wavfile.write(byte_data, rate=sampling_rate, data=speech_output[0].cpu().numpy())
    byte_data.seek(0)
    
    return StreamingResponse(byte_data, media_type="audio/wav")
