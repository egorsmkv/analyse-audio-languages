# Analyse used Languages in an audio file

## Idea

We want to analyse the languages used in an audio file.

Used models:

- MMS LID
- Whisper LID

It gives statistics in the following format:

**code_switched_speech.wav**:

```
la: 50.00%
jw: 43.75%
en: 6.25%
```

## Install

```
uv venv --python 3.12 && source .venv/bin/activate

uv pip install -r requirements.txt

# in development mode
uv pip install -r requirements-dev.txt
```

## Install VAD

Download the model from:

```
mkdir models

wget -O models/whisperx-vad-segmentation.bin "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"
```

## Download a code-switched speech

```
yt-dlp --extract-audio --audio-format wav -o "code_switched_speech.wav" https://www.youtube.com/watch?v=8mrys2qjhYs
```

## Run 

```
python analyse_mms_lid.py

python analyse_whisper_lid.py
```
