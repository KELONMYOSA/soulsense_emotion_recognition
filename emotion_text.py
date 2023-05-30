import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "cointegrated/rubert-tiny2-cedr-emotion-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)

id2label = {
    0: "neutral",
    1: "positive",
    2: "sad",
    3: "surprise",
    4: "fear",
    5: "angry"
}


def _predict(text):
    input_values = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(**input_values).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"label": id2label[i], "score": round(score, 5)} for i, score in enumerate(scores)]

    return outputs


def get_emotions(text):
    result = _predict(text)
    emotions = dict()
    for emotion in result:
        emotions[emotion["label"]] = emotion["score"]

    return emotions


def get_emotion(text):
    emotions = get_emotions(text)
    emotion = max(emotions, key=emotions.get)

    return emotion
