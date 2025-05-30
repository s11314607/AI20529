import os, base64, tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, pandas as pd, numpy as np, librosa
from pypinyin import lazy_pinyin

# 讀取你的 Gemini API Key
openai.api_key = os.getenv("AIzaSyDZIGKSH-7dQapglDfcxVU3ZsixHYc0Fq4")

# 載入阿美語資料庫（放在根目錄）
DB = pd.read_csv("ame_audio_database.csv")

app = Flask(__name__)
CORS(app)

def save_b64_wav(b64: str) -> str:
    _, b64_data = b64.split(",", 1)
    audio = base64.b64decode(b64_data)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(audio)
    return path

@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()
    if "audio_b64" not in data:
        return jsonify({"error": "缺少 audio_b64"}), 400

    wav_path = save_b64_wav(data["audio_b64"])
    try:
        # 1️⃣ 語音辨識（ASR）
        asr = openai.Audio.transcribe(
            file=open(wav_path, "rb"),
            model="whisper-1"
        )
        zh = asr["text"].strip()

        # 2️⃣ 資料庫查詢
        matched = DB[DB["zh"] == zh]
        if matched.empty:
            from difflib import get_close_matches
            cands = get_close_matches(zh, DB["zh"], n=1, cutoff=0.6)
            matched = DB[DB["zh"] == cands[0]] if cands else matched
        if matched.empty:
            return jsonify({"error": "查無對應阿美語"}), 404

        ame_text = matched.iloc[0]["ame_text"]
        ame_pinyin = matched.iloc[0]["ame_pinyin"]
        # 平均音調
        y, sr = librosa.load(matched.iloc[0]["ame_audio"], sr=None)
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        pitch = float(np.nanmean(f0[voiced])) if np.any(voiced) else None

        # 3️⃣ 聲音合成（TTS）
        tts = openai.Audio.generate(
            model="tts-1",
            voice="alloy",
            format="wav",
            input=ame_text
        )
        audio_b64 = tts["audio"]

        return jsonify({
            "zh": zh,
            "ame_text": ame_text,
            "ame_pinyin": ame_pinyin,
            "ame_pitch": pitch,
            "voice_clone_b64": f"data:audio/wav;base64,{audio_b64}"
        })
    finally:
        os.remove(wav_path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
