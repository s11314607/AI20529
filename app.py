from flask import Flask, request, jsonify, send_file
import tempfile, base64, os, datetime, shutil
import numpy as np, whisper, librosa, pandas as pd
from pypinyin import lazy_pinyin
from TTS.api import TTS

# 初始化模型（啟動時一次載入）
asr_small = whisper.load_model("small")
asr_large = whisper.load_model("large")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)

# 載入你的阿美語資料庫
DB = pd.read_csv("ame_audio_database.csv")  # 確保這支 CSV 放在同目錄

app = Flask(__name__)

def save_b64_wav(b64: str):
    data = base64.b64decode(b64.split(",",1)[1])
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f: f.write(data)
    return path

@app.route("/api/process", methods=["POST"])
def process():
    """
    POST JSON: { "audio_b64": "data:audio/wav;base64,...." }
    回傳 JSON: { zh: ..., ame_text:..., ame_pinyin:..., ame_pitch:..., tts_url:... }
    """
    data = request.json
    wav_path = save_b64_wav(data["audio_b64"])
    # 1. 語音辨識
    res = asr_small.transcribe(wav_path, beam_size=5, best_of=3, temperature=0.0)
    zh = res["text"].strip()

    # 2. 查庫
    match = DB[DB["zh"] == zh]
    if match.empty:
        # fuzzy
        from difflib import get_close_matches
        c = get_close_matches(zh, DB["zh"], n=1, cutoff=0.6)
        match = DB[DB["zh"] == c[0]] if c else match

    if match.empty:
        shutil.remove(wav_path)
        return jsonify({"error": "找不到對應的阿美語"}), 404

    ame_text = match.iloc[0]["ame_text"]
    ame_pinyin = match.iloc[0]["ame_pinyin"]
    # 3. 平均音調
    y, sr = librosa.load(match.iloc[0]["ame_audio"], sr=None)
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    ame_pitch = float(np.nanmean(f0[voiced])) if np.any(voiced) else None

    # 4. 聲音合成 (YourTTS)
    out_path = wav_path.replace(".wav", "_clone.wav")
    tts_model.tts_to_file(text=ame_text, speaker_wav=wav_path, language="en", file_path=out_path)

    # 5. 回傳並清理
    b64clone = "data:audio/wav;base64," + base64.b64encode(open(out_path, "rb").read()).decode()
    os.remove(wav_path)
    os.remove(out_path)

    return jsonify({
        "zh": zh,
        "ame_text": ame_text,
        "ame_pinyin": ame_pinyin,
        "ame_pitch": ame_pitch,
        "voice_clone_b64": b64clone
    })


if __name__ == "__main__":
    # 本機測試： flask run --host=0.0.0.0 --port=5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
