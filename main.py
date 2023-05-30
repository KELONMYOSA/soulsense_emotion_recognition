from fastapi import FastAPI, UploadFile

import emotion_audio
import emotion_text

app = FastAPI()


@app.post("/emotion-audio")
async def audio(file: UploadFile):
    return emotion_audio.get_emotion(file.file)


@app.post("/emotion-text")
async def text(text: str):
    return emotion_text.get_emotion(text)


@app.post("/emotion")
async def emotion(file: UploadFile, text: str):
    audio_emo = emotion_audio.get_emotions(file.file)
    text_emo = emotion_text.get_emotions(text)

    result_emo = {
        "neutral": (audio_emo["neutral"] + text_emo["neutral"]) / 2,
        "positive": (audio_emo["positive"] + text_emo["positive"]) / 2,
        "sad": (audio_emo["sad"] + text_emo["sad"]) / 2,
        "angry": (audio_emo["angry"] + text_emo["angry"]) / 2,
        "other": audio_emo["other"]
    }

    return max(result_emo, key=result_emo.get)
