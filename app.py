import os, base64, tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, pandas as pd, numpy as np, librosa
from pypinyin import lazy_pinyin
from werkzeug.utils import secure_filename
import pandas as pd
from pandas.errors import EmptyDataError

DB_CSV = "ame_audio_database.csv"
# 避免檔案不存在或是空檔
try:
    DB = pd.read_csv(DB_CSV)
except (FileNotFoundError, EmptyDataError):
    # 建一張空表並寫入欄位
    DB = pd.DataFrame(columns=["zh","ame_audio","ame_text","ame_pinyin","ame_pitch"])
    DB.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

# 接著既有的 Flask 路由就可以安全地使用 DB 了

# 讀取 Gemini API Key
openai.api_key = os.getenv("GEMINI_API_KEY")

# 載入阿美語資料庫
DB_CSV = "ame_audio_database.csv"
if os.path.exists(DB_CSV):
    DB = pd.read_csv(DB_CSV)
else:
    DB = pd.DataFrame(columns=["zh","ame_audio","ame_text","ame_pinyin","ame_pitch"])

app = Flask(__name__)
CORS(app)

# 儲存 base64 音檔至本地 wav
def save_b64_wav(b64: str) -> str:
    _, b64str = b64.split(',', 1)
    data = base64.b64decode(b64str)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return path

@app.route("/api/process", methods=["POST"])
def process():
    payload = request.get_json()
    wav_path = save_b64_wav(payload.get("audio_b64", ""))
    try:
        # ASR
        asr = openai.Audio.transcribe(file=open(wav_path, 'rb'), model="whisper-1")
        zh = asr['text'].strip()
        # 資料庫查詢
        match = DB[DB['zh'] == zh]
        if match.empty:
            from difflib import get_close_matches
            c = get_close_matches(zh, DB['zh'], n=1, cutoff=0.6)
            match = DB[DB['zh'] == c[0]] if c else match
        if match.empty:
            return jsonify({"error": "查無對應阿美語"}), 404
        row = match.iloc[0]
        ame_text = row['ame_text']
        ame_pinyin = row['ame_pinyin']
        # 計算音調
        y, sr = librosa.load(row['ame_audio'], sr=None)
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch = float(np.nanmean(f0[voiced])) if np.any(voiced) else None
        # TTS
        tts = openai.Audio.generate(model="tts-1", voice="alloy", format="wav", input=ame_text)
        b64out = tts['audio']
        return jsonify({
            "zh": zh,
            "ame_text": ame_text,
            "ame_pinyin": ame_pinyin,
            "ame_pitch": pitch,
            "voice_clone_b64": f"data:audio/wav;base64,{b64out}"
        })
    finally:
        os.remove(wav_path)

@app.route("/api/upload-db", methods=["POST"])
def upload_db():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "無檔案"}), 400
    new = []
    for f in files:
        fn = secure_filename(f.filename)
        dest = os.path.join("ame_audio", fn)
        os.makedirs("ame_audio", exist_ok=True)
        f.save(dest)
        zh = os.path.splitext(fn)[0]
        asr = openai.Audio.transcribe(file=open(dest,'rb'), model="whisper-1")
        ame = asr['text'].strip()
        ame_pinyin = " ".join(lazy_pinyin(ame if any(u'一'<=c<=u'龥' for c in ame) else zh))
        y, sr = librosa.load(dest, sr=None)
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch = float(np.nanmean(f0[voiced])) if np.any(voiced) else None
        new.append({"zh": zh, "ame_audio": dest, "ame_text": ame, "ame_pinyin": ame_pinyin, "ame_pitch": pitch})
    # 更新 CSV
    df = pd.concat([DB, pd.DataFrame(new)], ignore_index=True)
    df.to_csv(DB_CSV, index=False, encoding='utf-8-sig')
    return jsonify({"added": len(new), "entries": new})

if __name__ == "__main__":
    os.makedirs("ame_audio", exist_ok=True)
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
