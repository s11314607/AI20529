import os, tempfile, base64
import pandas as pd, numpy as np, librosa
from pypinyin import lazy_pinyin
import whisper
import openai
import gradio as gr

# 設定
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_CSV = "ame_audio_database.csv"
os.makedirs("ame_audio", exist_ok=True)

# 載入或初始化資料庫
try:
    DB = pd.read_csv(DB_CSV)
except:
    DB = pd.DataFrame(columns=["zh","ame_audio","ame_text","ame_pinyin","ame_pitch"])
    DB.to_csv(DB_CSV, index=False, encoding="utf-8-sig")

# 本機 Whisper
ASR = whisper.load_model("small")

def upload_db(files):
    new = []
    for f in files:
        # 存檔
        path = os.path.join("ame_audio", f.name)
        with open(path, "wb") as out: out.write(f.read())
        # ASR
        res = ASR.transcribe(path)
        ame = res["text"].strip()
        zh = os.path.splitext(f.name)[0]
        pyr = " ".join(lazy_pinyin(ame)) if any("\u4e00"<=c<="\u9fff" for c in ame) else " ".join(lazy_pinyin(zh))
        y, sr = librosa.load(path, sr=None)
        f0, v, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        pitch = float(np.nanmean(f0[v])) if np.any(v) else None
        new.append((zh, ame, pyr, pitch))
    df = pd.read_csv(DB_CSV)
    for zh,ame,pyr,p in new:
        df = df.append({"zh":zh,"ame_audio":path,"ame_text":ame,"ame_pinyin":pyr,"ame_pitch":p}, ignore_index=True)
    df.to_csv(DB_CSV,index=False,encoding="utf-8-sig")
    return f"新增 {len(new)} 筆"

def translate_and_tts(wav):
    # 存 wav
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd,"wb") as f: f.write(wav.read())
    # ASR 中文
    res = ASR.transcribe(path)
    zh = res["text"].strip()
    # 查庫
    df = pd.read_csv(DB_CSV)
    row = df[df["zh"]==zh]
    if row.empty: return "查無對應","", "", None
    ame = row.iloc[0]["ame_text"]
    pyr = row.iloc[0]["ame_pinyin"]
    # TTS
    tts = openai.Audio.generate(model="tts-1", voice="alloy", format="wav", input=ame)
    b64 = tts["audio"]
    return zh, ame, pyr, base64.b64decode(b64)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 中文→阿美語 翻譯 & 聲音克隆")
    with gr.Tab("更新母語資料庫"):
        db_uploader = gr.File(file_count="multiple")
        db_btn      = gr.Button("上傳並建立")
        db_out      = gr.Textbox()
        db_btn.click(upload_db, db_uploader, db_out)
    with gr.Tab("翻譯 & 合成"):
        wav_in  = gr.Audio(source="upload", type="file")
        btn     = gr.Button("執行")
        out1    = gr.Textbox(label="中文辨識")
        out2    = gr.Textbox(label="阿美語")
        out3    = gr.Textbox(label="拼音")
        out4    = gr.Audio(label="聲音克隆")
        btn.click(translate_and_tts, wav_in, [out1,out2,out3,out4])

demo.launch()
