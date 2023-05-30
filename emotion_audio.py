import logging

from transformers.pipelines import pipeline

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.CRITICAL)

pipe = pipeline(model="KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru", trust_remote_code=True)


def get_emotions(file):
    result = pipe(file)
    emotions = dict()
    for emotion in result:
        emotions[emotion["label"]] = emotion["score"]

    return emotions


def get_emotion(file):
    emotions = get_emotions(file)
    emotion = max(emotions, key=emotions.get)

    return emotion
