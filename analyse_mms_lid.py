import torch
import sphn

from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

from vad import load_vad_model, merge_chunks

device = torch.device("cuda:0")

filename = "code_switched_speech.wav"
chunk_size = 5.0
vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}

vad_model = load_vad_model(
    "models/whisperx-vad-segmentation.bin", device, **vad_options
)

lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/mms-lid-256"
).to(device)

audio_reader = sphn.FileReader(filename)

sr = audio_reader.sample_rate
audio = audio_reader.decode_all()

vad_segments = vad_model({"waveform": torch.tensor(audio), "sample_rate": sr})

vad_segments = merge_chunks(
    vad_segments,
    chunk_size,
    onset=vad_options["vad_onset"],
    offset=vad_options["vad_offset"],
)

def calculate_language_percentages(language_counts):
    total_count = sum(language_counts.values())
    language_percentages = {}
    for lang, count in language_counts.items():
        percentage = (count / total_count) * 100
        language_percentages[lang] = percentage
    return language_percentages

stats = {}
for idx, row in enumerate(vad_segments):
    seconds = row["end"] - row["start"]
    audio_data, _ = audio_reader.decode_with_padding(
        start_sec=row["start"], duration_sec=seconds
    )

    inputs = lid_processor(audio_data[0], sampling_rate=16_000, return_tensors="pt").to(
        device
    )

    with torch.inference_mode():
        outputs = lid_model(**inputs).logits

    lang_id = torch.argmax(outputs, dim=-1)[0].item()
    detected_lang = lid_model.config.id2label[lang_id]

    print(row)
    print(f"Detected language: {detected_lang}")

    if detected_lang not in stats:
        stats[detected_lang] = 1
    else:
        stats[detected_lang] += 1

print('---')
print(stats)
print('---')

percentages = calculate_language_percentages(stats)

for lang, percent in percentages.items():
    print(f"{lang}: {percent:.2f}%")
