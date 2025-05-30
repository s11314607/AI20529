import os, base64, tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, pandas as pd, numpy as np, librosa
from pypinyin import lazy_pinyin
from werkzeug.utils import secure_filename
from pandas.errors import EmptyDataError

# CSV 路徑
DB_CSV = "ame_audio_database.csv"
# 如果 CSV 不存在或是空檔，建立空 DataFrame
try:
    DB = pd.read_csv(DB_CSV)
except (FileNotFoundError, EmptyDataError):
    DB = pd.DataFrame(columns=["zh","ame_audio","ame_text","ame_pinyin","ame_pitch"])
    DB.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

# 讀取 API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

def save_b64_wav(b64: str) -> str:
    # 把前綴 data:…切掉
    _, b64str = b64.split(',', 1)
    data = base64.b64decode(b64str)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return path

@app.route("/api/process", methods=["POST"])
def process():
    payload = request.get_json(force=True)
    wav_path = save_b64_wav(payload.get("audio_b64",""))
    try:
        # 1. Whisper 辨識
        asr = openai.audio.transcriptions.create(
            file=open(wav_path, "rb"),
            model="whisper-1"
        )
        zh = asr["text"].strip()

        # 2. 資料庫查詢
        match = DB[DB["zh"] == zh]
        if match.empty:
            from difflib import get_close_matches
            cands = get_close_matches(zh, DB["zh"], n=1, cutoff=0.6)
            match = DB[DB["zh"] == cands[0]] if cands else match
        if match.empty:
            return jsonify({"error":"查無對應阿美語"}), 404
        row = match.iloc[0]

        # 3. 計算音調 (Hz)
        y, sr = librosa.load(row["ame_audio"], sr=None)
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        pitch = float(np.nanmean(f0[voiced])) if np.any(voiced) else None

        # 4. Gemini TTS（或 OpenAI TTS）
        speech = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=row["ame_text"],
            format="wav"
        )
        b64out = speech["audio"]

        return jsonify({
            "zh": zh,
            "ame_text": row["ame_text"],
            "ame_pinyin": row["ame_pinyin"],
            "ame_pitch": pitch,
            "voice_clone_b64": f"data:audio/wav;base64,{b64out}"
        })
    finally:
        os.remove(wav_path)

@app.route("/api/upload-db", methods=["POST"])
def upload_db():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error":"無檔案"}), 400

    new_entries = []
    for f in files:
        fn = secure_filename(f.filename)
        os.makedirs("ame_audio", exist_ok=True)
        dest = os.path.join("ame_audio", fn)
        f.save(dest)

        zh = os.path.splitext(fn)[0]
        asr = openai.audio.transcriptions.create(
            file=open(dest, "rb"),
            model="whisper-1"
        )
        ame = asr["text"].strip()
        ame_pinyin = " ".join(lazy_pinyin(ame if any(u"一"<=c<=u"龥" for c in ame) else zh))

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

    # Append 並寫回 CSV
    df_new = pd.DataFrame(new_entries)
    df = pd.concat([DB, df_new], ignore_index=True)
    df.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

    return jsonify({"added": len(new_entries), "entries": new_entries})

if __name__ == "__main__":
    os.makedirs("ame_audio", exist_ok=True)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
