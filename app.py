import os
import base64
import tempfile
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import librosa
from pypinyin import lazy_pinyin

import openai  # 改成直接 import openai

##### 1. 讀取 API key 並建立 client #####
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("請先於 Render Secrets 設定 OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)  # <- 用 openai.OpenAI()

##### 2. 初始化本地資料庫 #####
DB_CSV = "ame_audio_database.csv"
try:
    DB = pd.read_csv(DB_CSV)
except (FileNotFoundError, EmptyDataError):
    DB = pd.DataFrame(columns=["zh","ame_audio","ame_text","ame_pinyin","ame_pitch"])
    DB.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

app = Flask(__name__)
CORS(app)

def save_b64_wav(b64: str) -> str:
    """把 base64 wav 存成暫存檔，回傳檔案路徑"""
    _, b64str = b64.split(",", 1)
    data = base64.b64decode(b64str)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

@app.route("/api/process", methods=["POST"])
def process():
    payload = request.get_json(force=True)
    wav_path = save_b64_wav(payload.get("audio_b64", ""))
    try:
        # ASR
        asr = client.audio.transcriptions.create(
            file=open(wav_path, "rb"), 
            model="whisper-1"
        )
        zh = asr["text"].strip()

        # DB 查詢
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

        # 計算音調
        y, sr = librosa.load(row["ame_audio"], sr=None)
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        pitch = float(np.nanmean(f0[voiced])) if np.any(voiced) else None

        # TTS（Gemini TTS via OpenAI API）
        tts = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=ame_text,
            format="wav"
        )
        b64out = tts["audio"]

        return jsonify({
            "zh": zh,
            "ame_text": ame_text,
            "ame_pinyin": ame_pinyin,
            "ame_pitch": pitch,
            "voice_clone_b64": f"data:audio/wav;base64,{b64out}"
        })

    except Exception as e:
        print("=== process Exception ===")
        traceback.print_exc()
        return jsonify({"error":"伺服器錯誤","detail":str(e)}), 500
    finally:
        try: os.remove(wav_path)
        except: pass

@app.route("/api/upload-db", methods=["POST"])
def upload_db():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error":"沒有檔案"}), 400

    new_entries = []
    try:
        for f in files:
            fn = secure_filename(f.filename)
            os.makedirs("ame_audio", exist_ok=True)
            dest = os.path.join("ame_audio", fn)
            f.save(dest)

            zh = os.path.splitext(fn)[0]
            asr = client.audio.transcriptions.create(
                file=open(dest, "rb"),
                model="whisper-1"
            )
            ame = asr["text"].strip()

            # 拼音
            if any(u"\u4e00" <= c <= u"\u9fff" for c in ame):
                pinyin = " ".join(lazy_pinyin(ame))
            else:
                pinyin = " ".join(lazy_pinyin(zh))

            # pitch
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

        # 合併並存 CSV
        df_all = pd.concat([DB, pd.DataFrame(new_entries)], ignore_index=True)
        df_all.to_csv(DB_CSV, index=False, encoding="utf-8-sig")
        return jsonify({"added":len(new_entries), "entries":new_entries})

    except Exception as e:
        print("=== upload-db Exception ===")
        traceback.print_exc()
        return jsonify({"error":"更新失敗","detail":str(e)}), 500

if __name__ == "__main__":
    os.makedirs("ame_audio", exist_ok=True)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
