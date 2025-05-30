import os
import base64
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import openai
import pandas as pd
import numpy as np
import librosa
from pypinyin import lazy_pinyin
from pandas.errors import EmptyDataError

# --------------------------------------------------
# 設定：CSV 檔、資料夾路徑、OpenAI key
# --------------------------------------------------
DB_CSV = "ame_audio_database.csv"
AUDIO_DIR = "ame_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# 用舊版 openai SDK
openai.api_key = os.getenv("OPENAI_API_KEY")

# 讀取或初始化資料庫
try:
    DB = pd.read_csv(DB_CSV)
except (FileNotFoundError, EmptyDataError):
    DB = pd.DataFrame(columns=[
        "zh", "ame_audio", "ame_text", "ame_pinyin", "ame_pitch"
    ])
    DB.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__, static_folder=".", static_url_path="/")
CORS(app)

def save_b64_wav(b64: str) -> str:
    """把前端傳來的 base64 WAV 存成臨時檔案，回傳路徑"""
    _, b64str = b64.split(",", 1)
    data = base64.b64decode(b64str)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

@app.route("/")
def index():
    # 如果你也把 index.html 放在專案根目錄，就可以這樣
    return send_from_directory(".", "index.html")

@app.route("/api/process", methods=["POST"])
def process():
    payload = request.get_json()
    wav_path = save_b64_wav(payload.get("audio_b64", ""))
    try:
        # ----------------------
        # 1) Whisper 語音辨識
        # ----------------------
        asr = openai.Audio.transcribe(
            file=open(wav_path, "rb"),
            model="whisper-1"
        )
        zh = asr["text"].strip()

        # ----------------------
        # 2) 資料庫查詢
        # ----------------------
        match = DB[DB["zh"] == zh]
        if match.empty:
            from difflib import get_close_matches
            cand = get_close_matches(zh, DB["zh"], n=1, cutoff=0.6)
            if cand:
                match = DB[DB["zh"] == cand[0]]

        if match.empty:
            return jsonify({"error": "查無對應阿美語"}), 404

        row = match.iloc[0]
        ame_text   = row["ame_text"]
        ame_pinyin = row["ame_pinyin"]

        # ----------------------
        # 3) 計算音調（Hz）
        # ----------------------
        y, sr = librosa.load(row["ame_audio"], sr=None)
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        ame_pitch = float(np.nanmean(f0[voiced])) if np.any(voiced) else None

        # ----------------------
        # 4) Gemini TTS (via OpenAI API)
        # ----------------------
        # 回到舊版 openai SDK 的 TTS 呼叫
        tts = openai.Audio.generate(
            model="tts-1",
            voice="alloy",
            format="wav",
            input=ame_text
        )
        b64out = tts["audio"]

        return jsonify({
            "zh": zh,
            "ame_text": ame_text,
            "ame_pinyin": ame_pinyin,
            "ame_pitch": ame_pitch,
            "voice_clone_b64": f"data:audio/wav;base64,{b64out}"
        })
    finally:
        os.remove(wav_path)

@app.route("/api/upload-db", methods=["POST"])
def upload_db():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "無檔案"}), 400

    new_entries = []
    for f in files:
        fn = secure_filename(f.filename)
        dest = os.path.join(AUDIO_DIR, fn)
        f.save(dest)

        # 1) 原始檔名（不含副檔名）當中文句 zh
        zh = os.path.splitext(fn)[0]

        # 2) Whisper 辨識阿美語
        asr = openai.Audio.transcribe(
            file=open(dest, "rb"),
            model="whisper-1"
        )
        ame = asr["text"].strip()

        # 3) 拼音
        if any("\u4e00" <= ch <= "\u9fff" for ch in ame):
            ame_pinyin = " ".join(lazy_pinyin(ame))
        else:
            ame_pinyin = " ".join(lazy_pinyin(zh))

        # 4) 計算母語音檔音調
        y, sr = librosa.load(dest, sr=None)
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        pitch = float(np.nanmean(f0[voiced])) if np.any(voiced) else None

        new_entries.append({
            "zh": zh,
            "ame_audio": dest,
            "ame_text": ame,
            "ame_pinyin": ame_pinyin,
            "ame_pitch": pitch
        })

    # 把新資料 append 到舊的 DB
    df_all = pd.concat([DB, pd.DataFrame(new_entries)], ignore_index=True)
    df_all.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

    return jsonify({"added": len(new_entries), "entries": new_entries})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
