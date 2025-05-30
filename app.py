import os
import base64
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import pandas as pd
import numpy as np
import librosa
from pypinyin import lazy_pinyin
from werkzeug.utils import secure_filename
from pandas.errors import EmptyDataError

# ------------- 設定 & 載入環境變數 -------------
# API key from env
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("請在環境變數 OPENAI_API_KEY 中設定你的金鑰！")

# 資料庫 CSV 與語音檔資料夾
DB_CSV = "ame_audio_database.csv"
DB_AUDIO_FOLDER = "ame_audio"

# 確保資料夾存在
os.makedirs(DB_AUDIO_FOLDER, exist_ok=True)

# ------------- 讀取或初始化 CSV -------------
try:
    DB = pd.read_csv(DB_CSV)
except (FileNotFoundError, EmptyDataError):
    DB = pd.DataFrame(columns=["zh","ame_audio","ame_text","ame_pinyin","ame_pitch"])
    DB.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

# ------------- Flask App -------------
app = Flask(__name__)
CORS(app)

def save_b64_wav(b64: str) -> str:
    """把前端傳來的 base64 WAV 存成暫存檔，回傳路徑"""
    header, b64str = b64.split(",", 1)
    data = base64.b64decode(b64str)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

@app.route("/api/process", methods=["POST"])
def process():
    """處理前端的 翻譯＆聲音克隆 請求"""
    payload = request.get_json()
    b64 = payload.get("audio_b64")
    if not b64:
        return jsonify({"error":"請提供 audio_b64"}), 400

    wav_path = save_b64_wav(b64)
    try:
        # 1. Whisper ASR
        result = openai.Audio.transcriptions.create(
            file=open(wav_path, "rb"),
            model="whisper-1"
        )
        zh = result.text.strip()

        # 2. 資料庫查詢
        match = DB[DB["zh"] == zh]
        if match.empty:
            from difflib import get_close_matches
            cand = get_close_matches(zh, DB["zh"], n=1, cutoff=0.6)
            if cand:
                match = DB[DB["zh"] == cand[0]]
        if match.empty:
            return jsonify({"error":"查無對應阿美語"}), 404

        row = match.iloc[0]
        ame_text   = row["ame_text"]
        ame_pinyin = row["ame_pinyin"]

        # 3. TTS (Gemini)
        # 假設用 openai.ChatCompletion + voice parameter
        tts_resp = openai.Audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=ame_text,
            format="wav"
        )
        b64_audio = tts_resp.audio  # 已經是 base64 字串

        return jsonify({
            "zh": zh,
            "ame_text": ame_text,
            "ame_pinyin": ame_pinyin,
            "voice_clone_b64": f"data:audio/wav;base64,{b64_audio}"
        })

    finally:
        os.remove(wav_path)

@app.route("/api/upload-db", methods=["POST"])
def upload_db():
    """處理前端的 上傳母語資料庫 請求"""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error":"請上傳至少一個母語音檔"}), 400

    new_entries = []
    for f in files:
        filename = secure_filename(f.filename)
        dest = os.path.join(DB_AUDIO_FOLDER, filename)
        f.save(dest)

        # 1. 提取檔名當中文
        zh = os.path.splitext(filename)[0]

        # 2. Whisper ASR
        asr_res = openai.Audio.transcriptions.create(
            file=open(dest, "rb"),
            model="whisper-1"
        )
        ame_text = asr_res.text.strip()

        # 3. 拼音
        if any("\u4e00" <= c <= "\u9fff" for c in ame_text):
            ame_pinyin = " ".join(lazy_pinyin(ame_text))
        else:
            ame_pinyin = " ".join(lazy_pinyin(zh))

        # 4. 計算音調 (僅示範，不回傳給前端)
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
            "ame_text": ame_text,
            "ame_pinyin": ame_pinyin,
            "ame_pitch": pitch
        })

    # 5. 寫回 CSV
    global DB
    DB = pd.concat([DB, pd.DataFrame(new_entries)], ignore_index=True)
    DB.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

    return jsonify({
        "added": len(new_entries),
        "entries": new_entries
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
