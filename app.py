import os
import base64
import tempfile
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import whisper
import openai
import pandas as pd
import numpy as np
import librosa
from pypinyin import lazy_pinyin
from pandas.errors import EmptyDataError

# ----------------------
# 設定區
# ----------------------
DB_CSV      = "ame_audio_database.csv"
AUDIO_DIR   = "ame_audio"
PORT        = int(os.getenv("PORT", 5000))
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError("請設定環境變數 OPENAI_API_KEY")

os.makedirs(AUDIO_DIR, exist_ok=True)

# 舊版 openai TTS
openai.api_key = OPENAI_KEY

# 載入或初始化本地 CSV 資料庫
try:
    DB = pd.read_csv(DB_CSV)
except (FileNotFoundError, EmptyDataError):
    DB = pd.DataFrame(columns=[
        "zh","ame_audio","ame_text","ame_pinyin","ame_pitch"
    ])
    DB.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

# 載入本機 Whisper 模型（large 或你選的）
ASR_MODEL = whisper.load_model("large")

# ----------------------
# Flask App
# ----------------------
app = Flask(__name__, static_folder=".", static_url_path="/")
CORS(app)

def save_b64_wav(b64: str) -> str:
    """把 base64 wav 存成臨時檔，回傳路徑"""
    _, b64str = b64.split(",", 1)
    data = base64.b64decode(b64str)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/upload-db", methods=["POST"])
def upload_db():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error":"請上傳至少一個音檔"}), 400

    new_entries = []
    try:
        for f in files:
            fn = secure_filename(f.filename)
            dest = os.path.join(AUDIO_DIR, fn)
            f.save(dest)

            # 1️⃣ 用本機 Whisper 辨識母語文字
            res = ASR_MODEL.transcribe(
                dest,
                beam_size=5, best_of=3, temperature=0.0
            )
            ame = res["text"].strip()

            # 2️⃣ 檔名（不含副檔名）當作中文 zh
            zh = os.path.splitext(fn)[0]

            # 3️⃣ 轉拼音
            if any("\u4e00" <= ch <= "\u9fff" for ch in ame):
                pinyin = " ".join(lazy_pinyin(ame))
            else:
                pinyin = " ".join(lazy_pinyin(zh))

            # 4️⃣ 計算音調（Hz）
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
                "ame_pinyin": pinyin,
                "ame_pitch": pitch
            })

        # 5️⃣ 更新 CSV
        df_all = pd.concat([DB, pd.DataFrame(new_entries)], ignore_index=True)
        df_all.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

        return jsonify({"added": len(new_entries), "entries": new_entries})

    except Exception as e:
        print("=== upload-db Exception ===")
        traceback.print_exc()
        return jsonify({"error":"更新資料庫失敗","detail":str(e)}), 500

@app.route("/api/process", methods=["POST"])
def process():
    payload = request.get_json(force=True)
    b64 = payload.get("audio_b64", "")
    if not b64:
        return jsonify({"error":"缺少 audio_b64"}), 400

    wav_path = save_b64_wav(b64)
    try:
        # 1️⃣ 本機 Whisper 辨識中文
        res = ASR_MODEL.transcribe(
            wav_path,
            beam_size=5, best_of=3, temperature=0.0
        )
        zh = res["text"].strip()

        # 2️⃣ 查 CSV
        match = DB[DB["zh"] == zh]
        if match.empty:
            from difflib import get_close_matches
            cands = get_close_matches(zh, DB["zh"], n=1, cutoff=0.6)
            if cands:
                match = DB[DB["zh"] == cands[0]]
        if match.empty:
            return jsonify({"error":"查無對應阿美語"}), 404

        row = match.iloc[0]
        ame_text   = row["ame_text"]
        ame_pinyin = row["ame_pinyin"]

        # 3️⃣ TTS：OpenAI / Gemini
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
            "ame_pitch": row["ame_pitch"],
            "voice_clone_b64": f"data:audio/wav;base64,{b64out}"
        })

    except Exception as e:
        print("=== process Exception ===")
        traceback.print_exc()
        return jsonify({"error":"合成失敗","detail":str(e)}), 500

    finally:
        try: os.remove(wav_path)
        except: pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
