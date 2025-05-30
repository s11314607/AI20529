import os, base64, tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

# 讀取你的 Gemini API Key
openai.api_key = os.getenv("AIzaSyDZIGKSH-7dQapglDfcxVU3ZsixHYc0Fq4")

app = Flask(__name__)
CORS(app)  # 允許前端跨域呼叫

@app.route("/api/process", methods=["POST"])
def process_audio():
    data = request.get_json()
    audio_b64 = data.get("audio_b64")
    if not audio_b64:
        return jsonify({"error": "缺少 audio_b64"}), 400

    # 1. 解 base64 並存成暫存檔
    header, b64 = audio_b64.split(",", 1)
    audio_bytes = base64.b64decode(b64)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)

    try:
        # 2. 呼叫 Gemini ASR (語音轉文字)
        asr_resp = openai.Audio.transcribe(
            file=open(wav_path, "rb"),
            model="whisper-1"  # 或 Gemni ASR model 名稱
        )
        zh_text = asr_resp["text"].strip()

        # >>> 在這裡如果你還要查阿美語資料庫，就插入那段邏輯 <<<

        # 3. 呼叫 Gemini TTS (文字轉語音)
        tts_resp = openai.Audio.generate(
            model="tts-1",       # Gemini TTS model
            voice="alloy",       # 可用 voice ID
            format="wav",
            input=zh_text        # 這邊示範把同一句中文再合回去
        )
        audio_out_b64 = tts_resp["audio"]  # Gemini 回傳的 base64

        return jsonify({
            "zh": zh_text,
            "tts_audio_b64": f"data:audio/wav;base64,{audio_out_b64}"
        })

    finally:
        # 清理暫存檔
        os.remove(wav_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
