# ============================================================
# Repetition.py (機械音声の記号（パイプ等）読み上げ防止・完全版)
# ============================================================

import csv
import datetime
import difflib
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import parselmouth

from janome.tokenizer import Tokenizer
import pykakasi

# pykakasiの初期化（漢字・カタカナのひらがな変換用）
kks = pykakasi.kakasi()


# ============================================================
# 基本設定
# ============================================================

TASK_NAME = "Repetition"  # フォルダ名に使用する課題名
FS = 44100
DEFAULT_WHISPER_MODEL_NAME = "turbo"
DEFAULT_MAX_DURATION = 15  # デフォルト制限時間：15秒

# RMS がこの値より低い録音は無反応・ほぼ無音として扱う
SILENCE_RMS_THRESHOLD = 0.002

# 音声検出用のしきい値
VOICE_THRESHOLD = 0.08


# ============================================================
# パス設定
# ============================================================

def get_app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = get_app_dir()
SL_TOOL_DIR = os.path.dirname(APP_DIR)

STIMULI_AUDIO_FOLDER = os.path.join(SL_TOOL_DIR, "stimuli", "Repetition")
RESULTS_DIR = os.path.join(SL_TOOL_DIR, "results")
MODEL_DIR = os.path.join(SL_TOOL_DIR, "whisper_model")

os.environ["PATH"] = SL_TOOL_DIR + os.pathsep + os.environ.get("PATH", "")


# ============================================================
# 機械音声（Windows TTS）再生ヘルパー関数（パイプ等の記号を除去）
# ============================================================
def speak_text_tts(text):
    try:
        # 「｜」「|」「／」「/」などの記号を削除・置換して読み上げを防ぐ
        clean_text = re.sub(r'[｜|／/]', '', text)
        escaped = clean_text.replace("'", "''")
        cmd = (
            f'powershell -Command "Add-Type -AssemblyName System.Speech; '
            f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$synth.Volume = 100; $synth.Speak(\'{escaped}\')"'
        )
        subprocess.run(cmd, shell=True)
    except Exception as e:
        print(f"[警告] 機械音声の再生に失敗しました: {e}")


# ============================================================
# 被験者情報（ID・名前）入力 GUI
# ============================================================

def input_participant_info():
    root = tk.Tk()
    root.title("被験者情報の入力")
    root.configure(bg="lightblue")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    id_var = tk.StringVar(value="1")
    name_var = tk.StringVar(value="")
    result = [None]

    container = tk.Frame(root, bg="lightblue")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【被験者情報の入力】",
        font=("Meiryo", 32, "bold"),
        bg="lightblue",
        fg="black"
    ).pack(pady=(0, 15))

    tk.Label(
        container,
        text="被験者IDと名前（イニシャル等）を入力してください。\n（Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightblue",
        fg="darkblue",
        justify="center"
    ).pack(pady=(0, 30))

    form_frame = tk.Frame(container, bg="white", padx=50, pady=35)
    form_frame.pack(pady=10)

    row1 = tk.Frame(form_frame, bg="white")
    row1.pack(fill="x", pady=12)
    tk.Label(row1, text="被験者 ID :", font=("Meiryo", 20, "bold"), bg="white", width=12, anchor="e").pack(side="left", padx=(0, 15))
    entry_id = tk.Entry(row1, textvariable=id_var, font=("Meiryo", 20), width=18)
    entry_id.pack(side="left")

    row2 = tk.Frame(form_frame, bg="white")
    row2.pack(fill="x", pady=12)
    tk.Label(row2, text="名前 :", font=("Meiryo", 20, "bold"), bg="white", width=12, anchor="e").pack(side="left", padx=(0, 15))
    entry_name = tk.Entry(row2, textvariable=name_var, font=("Meiryo", 20), width=18)
    entry_name.pack(side="left")

    button_frame = tk.Frame(container, bg="lightblue")
    button_frame.pack(pady=40)

    def submit(event=None):
        p_id = id_var.get().strip()
        p_name = name_var.get().strip()
        if not p_id or not p_name:
            messagebox.showerror("入力エラー", "被験者IDと名前の両方を入力してください。")
            return
        safe_id = re.sub(r'[\\/*?:"<>|]', "", p_id)
        safe_name = re.sub(r'[\\/*?:"<>|]', "", p_name)
        result[0] = (safe_id, safe_name)
        try:
            root.destroy()
        except Exception:
            pass

    def cancel(event=None):
        result[0] = None
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.bind_all("<Return>", submit)
    root.bind_all("<Escape>", cancel)

    tk.Button(
        button_frame,
        text="決定して次へ (Enter)",
        command=submit,
        font=("Meiryo", 16, "bold"),
        width=24,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    tk.Button(
        button_frame,
        text="キャンセル / 終了 (Esc)",
        command=cancel,
        font=("Meiryo", 16),
        width=20,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    entry_name.focus_set()
    root.lift()
    root.focus_force()
    root.mainloop()

    return result[0]


# ============================================================
# stimuli/Repetition 内の利用可能なソースを取得
# ============================================================
def find_available_stimulus_sources():
    sources = []

    if not os.path.exists(STIMULI_AUDIO_FOLDER):
        os.makedirs(STIMULI_AUDIO_FOLDER, exist_ok=True)

    wav_files = [f for f in os.listdir(STIMULI_AUDIO_FOLDER) if f.lower().endswith(".wav")]
    sources.append({
        "type": "wav",
        "label": f"音声ファイル (.wav) を使用（stimuli/Repetition 内: {len(wav_files)}件）",
        "path": STIMULI_AUDIO_FOLDER
    })

    txt_files = sorted([f for f in os.listdir(STIMULI_AUDIO_FOLDER) if f.lower().endswith(".txt")])
    for f in txt_files:
        full_p = os.path.join(STIMULI_AUDIO_FOLDER, f)
        sources.append({
            "type": "txt",
            "label": f"テキストファイル: {f} （機械音声で読み上げ）",
            "path": full_p
        })

    return sources


# ============================================================
# 刺激ソース選択 GUI
# ============================================================
def select_stimulus_source_gui():
    sources = find_available_stimulus_sources()

    root = tk.Tk()
    root.title("刺激ソースの選択")
    root.configure(bg="lightblue")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    selected_idx = tk.IntVar(value=0)
    result = [None]

    container = tk.Frame(root, bg="lightblue")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【刺激データ（WAV / TXT）の選択】",
        font=("Meiryo", 32, "bold"),
        bg="lightblue",
        fg="black"
    ).pack(pady=(0, 15))

    tk.Label(
        container,
        text="stimuli/Repetition 内のどの刺激を使用するか選択してください。\n（上下キーまたはマウスで選択し、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightblue",
        fg="darkblue",
        justify="center"
    ).pack(pady=(0, 30))

    list_frame = tk.Frame(container, bg="white", padx=40, pady=25)
    list_frame.pack(pady=10)

    for idx, src in enumerate(sources):
        tk.Radiobutton(
            list_frame,
            text=src["label"],
            variable=selected_idx,
            value=idx,
            font=("Meiryo", 16, "bold"),
            bg="white",
            activebackground="white",
            anchor="w"
        ).pack(fill="x", pady=8)

    button_frame = tk.Frame(container, bg="lightblue")
    button_frame.pack(pady=40)

    def confirm_source(event=None):
        result[0] = sources[selected_idx.get()]
        try:
            root.destroy()
        except Exception:
            pass

    def cancel_source(event=None):
        result[0] = None
        try:
            root.destroy()
        except Exception:
            pass

    def move_selection(direction):
        new_idx = (selected_idx.get() + direction) % len(sources)
        selected_idx.set(new_idx)

    root.bind_all("<Up>", lambda e: move_selection(-1))
    root.bind_all("<Down>", lambda e: move_selection(1))
    root.bind_all("<Return>", confirm_source)
    root.bind_all("<Escape>", cancel_source)
    root.protocol("WM_DELETE_WINDOW", cancel_source)

    tk.Button(
        button_frame,
        text="決定して項目選択へ (Enter)",
        command=confirm_source,
        font=("Meiryo", 16, "bold"),
        width=26,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    tk.Button(
        button_frame,
        text="キャンセル / 終了 (Esc)",
        command=cancel_source,
        font=("Meiryo", 16),
        width=20,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    root.lift()
    root.focus_force()
    root.mainloop()

    return result[0]


# ============================================================
# WAVファイル読み込み関数
# ============================================================
def load_stimuli_from_audio_folder(folder_path=STIMULI_AUDIO_FOLDER):
    stimuli_list = []

    if os.path.exists(folder_path):
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]

        def extract_number(filename):
            match = re.match(r'^(\d+)_', filename)
            return int(match.group(1)) if match else 9999

        wav_files.sort(key=extract_number)

        for f in wav_files:
            name_without_ext = os.path.splitext(f)[0]
            parts = name_without_ext.split('_', 1)
            if len(parts) == 2:
                stimulus_text = parts[1].strip()
                audio_path = os.path.join(folder_path, f)
                stimuli_list.append((stimulus_text, audio_path))

    if not stimuli_list:
        stimuli_list = [
            ("でんしゃ", None),
            ("おとこの人が歩いている", None)
        ]

    return stimuli_list


# ============================================================
# TXTファイル読み込み関数
# ============================================================
def load_stimuli_from_txt(txt_path):
    stimuli_list = []
    if not os.path.exists(txt_path):
        return stimuli_list

    lines = []
    for enc in ["utf-8-sig", "utf-8", "cp932"]:
        try:
            with open(txt_path, mode="r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    parsed_items = []
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = re.match(r'^(\d+)_(.+)$', line)
        if match:
            num = int(match.group(1))
            stimulus_text = match.group(2).strip()
        else:
            num = idx
            stimulus_text = line

        if stimulus_text:
            parsed_items.append((num, stimulus_text))

    parsed_items.sort(key=lambda x: x[0])

    for _, stimulus_text in parsed_items:
        stimuli_list.append((stimulus_text, None))

    return stimuli_list


# ============================================================
# 個別刺激の選択 GUI
# ============================================================
def select_stimuli_gui(all_stimuli, source_type="wav"):
    root = tk.Tk()
    root.title("実施刺激の選択")
    root.configure(bg="lightblue")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    selected_result = []
    cancelled = [False]

    container = tk.Frame(root, bg="lightblue")
    container.pack(expand=True, fill="both", padx=40, pady=30)

    mode_str = "WAV音声ファイル" if source_type == "wav" else "TXTテキスト（機械音声）"
    tk.Label(
        container,
        text=f"【課題に使用する刺激の選択（{mode_str}）】",
        font=("Meiryo", 26, "bold"),
        bg="lightblue",
        fg="black"
    ).pack(pady=(0, 10))

    tk.Label(
        container,
        text="実施する項目にチェックを入れてください。（「試聴」ボタンで提示音声を事前に確認できます / Escキーで終了）",
        font=("Meiryo", 14),
        bg="lightblue",
        fg="darkblue",
        justify="center"
    ).pack(pady=(0, 15))

    canvas_frame = tk.Frame(container, bg="white")
    canvas_frame.pack(fill="both", expand=True, padx=20, pady=10)

    canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
    scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="white")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    vars_list = []

    def preview_stimulus(text, path):
        if path and os.path.exists(path):
            try:
                data, file_fs = sf.read(path, dtype="float32")
                sd.play(data, file_fs)
            except Exception as e:
                print(f"[警告] 試聴再生エラー: {e}")
        else:
            threading.Thread(target=speak_text_tts, args=(text,), daemon=True).start()

    for idx, (stim_text, audio_path) in enumerate(all_stimuli, start=1):
        var = tk.BooleanVar(value=True)
        vars_list.append((var, stim_text, audio_path))

        row_f = tk.Frame(scrollable_frame, bg="white", pady=5)
        row_f.pack(fill="x", padx=10, anchor="w")

        source_desc = os.path.basename(audio_path) if audio_path else "機械音声(TTS)"
        chk = tk.Checkbutton(
            row_f,
            text=f" {idx}. {stim_text}  [{source_desc}]",
            variable=var,
            font=("Meiryo", 14),
            bg="white",
            activebackground="white",
            anchor="w"
        )
        chk.pack(side="left", padx=5)

        btn_preview = tk.Button(
            row_f,
            text="試聴",
            font=("Meiryo", 11),
            command=lambda t=stim_text, p=audio_path: preview_stimulus(t, p),
            width=6,
            bg="#e0e0e0"
        )
        btn_preview.pack(side="right", padx=10)

    ctrl_frame = tk.Frame(container, bg="lightblue")
    ctrl_frame.pack(pady=10)

    def select_all(val):
        for v, _, _ in vars_list:
            v.set(val)

    tk.Button(ctrl_frame, text="全選択", command=lambda: select_all(True), font=("Meiryo", 12), width=12, bg="white").pack(side="left", padx=10)
    tk.Button(ctrl_frame, text="全解除", command=lambda: select_all(False), font=("Meiryo", 12), width=12, bg="white").pack(side="left", padx=10)

    button_frame = tk.Frame(container, bg="lightblue")
    button_frame.pack(pady=15)

    def confirm_selection(event=None):
        chosen = [(text, path) for v, text, path in vars_list if v.get()]
        if not chosen:
            messagebox.showerror("選択エラー", "1つ以上の刺激を選択してください。")
            return
        selected_result.extend(chosen)
        try:
            root.destroy()
        except Exception:
            pass

    def cancel_selection(event=None):
        cancelled[0] = True
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", cancel_selection)
    root.bind_all("<Return>", confirm_selection)
    root.bind_all("<Escape>", cancel_selection)

    tk.Button(
        button_frame,
        text="この選択で開始 (Enter)",
        command=confirm_selection,
        font=("Meiryo", 16, "bold"),
        width=24,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    tk.Button(
        button_frame,
        text="キャンセル / 終了 (Esc)",
        command=cancel_selection,
        font=("Meiryo", 16),
        width=20,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    root.lift()
    root.focus_force()
    root.mainloop()

    if cancelled[0]:
        return None
    return selected_result


# ============================================================
# テキストの正規化・ひらがな読み変換関数
# ============================================================
def normalize_text_for_evaluation(text):
    if not text:
        return ""
    clean = re.sub(r'[、。！？\s｜|／/]', '', text)
    try:
        result = kks.convert(clean)
        return "".join([item.get('hira', '') for item in result])
    except Exception:
        return clean


# ============================================================
# ひらがなテキストのモーラ（拍）リスト分解関数
# ============================================================
def get_mora_list(hira_text):
    moras = []
    combine_chars = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
    i = 0
    while i < len(hira_text):
        char = hira_text[i]
        if i + 1 < len(hira_text) and hira_text[i + 1] in combine_chars:
            moras.append(char + hira_text[i + 1])
            i += 2
        else:
            moras.append(char)
            i += 1
    return moras


# ============================================================
# モーラ数（拍数）の計算関数
# ============================================================
def count_moras(text):
    clean_text = text.replace("｜", "").replace("|", "").replace("／", "").replace("/", "")
    hira_text = normalize_text_for_evaluation(clean_text)
    moras = get_mora_list(hira_text)
    return max(1, len(moras))


# ============================================================
# 構音の歪み評価 ＆ フォルマント付きスペクトログラム生成関数
# ============================================================
def analyze_and_plot_formants(audio_path, output_image_path):
    if not os.path.exists(audio_path):
        return 0.0, ""

    try:
        y, sr = librosa.load(audio_path, sr=FS)
        if len(y) == 0:
            return 0.0, ""

        y_trimmed, index = librosa.effects.trim(y, top_db=30)
        if len(y_trimmed) == 0:
            y_trimmed = y

        stft = np.abs(librosa.stft(y_trimmed))
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_mask = freqs >= 3000
        total_energy = np.sum(stft)
        high_energy = np.sum(stft[high_freq_mask, :])
        high_freq_ratio = round((high_energy / (total_energy + 1e-10)) * 100, 2)

        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.figure(figsize=(10, 4))
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis='time', y_axis='hz',
            cmap='viridis', vmin=-80, vmax=0
        )
        plt.ylim(0, 8000)

        snd = parselmouth.Sound(audio_path)
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)

        time_steps = formant.xs()
        f1_list, f2_list, f3_list = [], [], []

        for t in time_steps:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)

            f1_list.append(f1 if (f1 is not None and not np.isnan(f1)) else np.nan)
            f2_list.append(f2 if (f2 is not None and not np.isnan(f2)) else np.nan)
            f3_list.append(f3 if (f3 is not None and not np.isnan(f3)) else np.nan)

        plt.scatter(time_steps, f1_list, color='red', s=4, label='F1')
        plt.scatter(time_steps, f2_list, color='blue', s=4, label='F2')
        plt.scatter(time_steps, f3_list, color='green', s=4, label='F3')

        plt.colorbar(img, format='%+2.0f dB')
        plt.title(f'Speech Spectrogram with Formants (High-Freq Ratio: {high_freq_ratio}%)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close()

        return high_freq_ratio, os.path.basename(output_image_path)
    except Exception as e:
        print(f"[警告] フォルマント分析エラー: {e}")
        return 0.0, ""


# ============================================================
# 効果音（クリック音）生成関数
# ============================================================

def generate_click_sound(fs=FS, duration=0.03, freq=1000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-15 * t / duration)
    return audio.astype(np.float32)


# ============================================================
# Whisper モデル選択 GUI
# ============================================================

def select_model():
    root = tk.Tk()
    root.title("Whisper モデル選択")
    root.configure(bg="lightblue")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    selected_model = tk.StringVar(value=DEFAULT_WHISPER_MODEL_NAME)
    selection_result = [None]

    container = tk.Frame(root, bg="lightblue")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【Whisper 音声認識モデルの選択】",
        font=("Meiryo", 32, "bold"),
        bg="lightblue",
        fg="black"
    ).pack(pady=(0, 15))

    tk.Label(
        container,
        text="文字起こしに使用するモデルを選択してください。\n（上下キーで選択変更、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightblue",
        fg="darkblue",
        justify="center"
    ).pack(pady=(0, 30))

    model_frame = tk.Frame(container, bg="white", padx=40, pady=30)
    model_frame.pack(pady=10)

    model_options = [
        ("small", "Small", "速度と精度のバランスを重視"),
        ("turbo", "Turbo", "高速処理向け（初期選択）"),
        ("large", "Large", "高精度、処理時間・必要メモリは大きい")
    ]

    for model_name, label, description in model_options:
        option_frame = tk.Frame(model_frame, bg="white")
        option_frame.pack(fill="x", pady=10)

        tk.Radiobutton(
            option_frame,
            text=label,
            variable=selected_model,
            value=model_name,
            font=("Meiryo", 18, "bold"),
            bg="white",
            activebackground="white",
            width=12,
            anchor="w"
        ).pack(side="left", padx=(0, 20))

        tk.Label(
            option_frame,
            text=description,
            font=("Meiryo", 14),
            bg="white",
            fg="gray"
        ).pack(side="left")

    button_frame = tk.Frame(container, bg="lightblue")
    button_frame.pack(pady=40)

    def start_selected_model(event=None):
        selection_result[0] = selected_model.get()
        try:
            root.destroy()
        except Exception:
            pass

    def cancel_selection(event=None):
        selection_result[0] = None
        try:
            root.destroy()
        except Exception:
            pass

    model_values = [opt[0] for opt in model_options]

    def move_selection(direction):
        current_val = selected_model.get()
        try:
            current_idx = model_values.index(current_val)
        except ValueError:
            current_idx = 1
        new_idx = (current_idx + direction) % len(model_values)
        selected_model.set(model_values[new_idx])

    root.bind_all("<Up>", lambda event: move_selection(-1))
    root.bind_all("<Down>", lambda event: move_selection(1))

    tk.Button(
        button_frame,
        text="このモデルで開始 (Enter)",
        command=start_selected_model,
        font=("Meiryo", 16, "bold"),
        width=24,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    tk.Button(
        button_frame,
        text="キャンセル / 終了 (Esc)",
        command=cancel_selection,
        font=("Meiryo", 16),
        width=20,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    root.protocol("WM_DELETE_WINDOW", cancel_selection)
    root.bind_all("<Return>", start_selected_model)
    root.bind_all("<Escape>", cancel_selection)

    root.lift()
    root.focus_force()
    root.focus_set()
    root.mainloop()

    return selection_result[0]


# ============================================================
# 実験モード選択 GUI
# ============================================================

def select_experiment_mode():
    root = tk.Tk()
    root.title("実験モード選択")
    root.configure(bg="lightblue")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    selected_mode = tk.BooleanVar(value=True)
    selection_result = [None]

    container = tk.Frame(root, bg="lightblue")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【実験モードの選択】",
        font=("Meiryo", 32, "bold"),
        bg="lightblue",
        fg="black"
    ).pack(pady=(0, 15))

    tk.Label(
        container,
        text="ヒント機能の有無を選択してください。\n（上下キーで選択変更、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightblue",
        fg="darkblue",
        justify="center"
    ).pack(pady=(0, 30))

    mode_frame = tk.Frame(container, bg="white", padx=40, pady=30)
    mode_frame.pack(pady=10)

    options = [
        (True, "ヒントあり", "誤答時に刺激音声を再再生し、再録音を実施します（標準）"),
        (False, "ヒントなし", "各試行1回のみの録音で測定を完了します")
    ]

    for val, label, desc in options:
        option_frame = tk.Frame(mode_frame, bg="white")
        option_frame.pack(fill="x", pady=10)

        tk.Radiobutton(
            option_frame,
            text=label,
            variable=selected_mode,
            value=val,
            font=("Meiryo", 18, "bold"),
            bg="white",
            activebackground="white",
            anchor="w"
        ).pack(anchor="w", padx=10, pady=(0, 2))

        tk.Label(
            option_frame,
            text=desc,
            font=("Meiryo", 14),
            bg="white",
            fg="gray"
        ).pack(anchor="w", padx=35)

    button_frame = tk.Frame(container, bg="lightblue")
    button_frame.pack(pady=40)

    def start_mode(event=None):
        selection_result[0] = selected_mode.get()
        try:
            root.destroy()
        except Exception:
            pass

    def cancel_mode(event=None):
        selection_result[0] = None
        try:
            root.destroy()
        except Exception:
            pass

    mode_values = [opt[0] for opt in options]

    def move_mode_selection(direction):
        current_val = selected_mode.get()
        try:
            current_idx = mode_values.index(current_val)
        except ValueError:
            current_idx = 0
        new_idx = (current_idx + direction) % len(mode_values)
        selected_mode.set(mode_values[new_idx])

    root.bind_all("<Up>", lambda event: move_mode_selection(-1))
    root.bind_all("<Down>", lambda event: move_mode_selection(1))

    tk.Button(
        button_frame,
        text="このモードで開始 (Enter)",
        command=start_mode,
        font=("Meiryo", 16, "bold"),
        width=24,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    tk.Button(
        button_frame,
        text="キャンセル / 終了 (Esc)",
        command=cancel_mode,
        font=("Meiryo", 16),
        width=20,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    root.protocol("WM_DELETE_WINDOW", cancel_mode)
    root.bind_all("<Return>", start_mode)
    root.bind_all("<Escape>", cancel_mode)

    root.lift()
    root.focus_force()
    root.focus_set()
    root.mainloop()

    return selection_result[0]


# ============================================================
# 制限時間設定 GUI
# ============================================================

def select_duration_setting():
    root = tk.Tk()
    root.title("制限時間設定")
    root.configure(bg="lightblue")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    duration_var = tk.StringVar(value=str(DEFAULT_MAX_DURATION))
    selection_result = [None]

    container = tk.Frame(root, bg="lightblue")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【課題制限時間の設定】",
        font=("Meiryo", 32, "bold"),
        bg="lightblue",
        fg="black"
    ).pack(pady=(0, 15))

    tk.Label(
        container,
        text="1試行あたりの制限時間（秒）を半角数字で入力してください。\n（指定時間を超えると自動的に録音が終了します / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightblue",
        fg="darkblue",
        justify="center"
    ).pack(pady=(0, 30))

    input_frame = tk.Frame(container, bg="white", padx=50, pady=30)
    input_frame.pack(pady=10)

    tk.Entry(
        input_frame,
        textvariable=duration_var,
        font=("Meiryo", 24),
        width=8,
        justify="center"
    ).pack(side=tk.LEFT, padx=(0, 15))

    tk.Label(
        input_frame,
        text="秒",
        font=("Meiryo", 22, "bold"),
        bg="white"
    ).pack(side=tk.LEFT)

    button_frame = tk.Frame(container, bg="lightblue")
    button_frame.pack(pady=40)

    def start_duration(event=None):
        try:
            val = int(duration_var.get().strip())
            if val <= 0:
                raise ValueError
            selection_result[0] = val
            try:
                root.destroy()
            except Exception:
                pass
        except ValueError:
            messagebox.showerror("入力エラー", "制限時間は正の整数で入力してください。")

    def cancel_duration(event=None):
        selection_result[0] = None
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", cancel_duration)
    root.bind_all("<Return>", start_duration)
    root.bind_all("<Escape>", cancel_duration)

    tk.Button(
        button_frame,
        text="決定して次へ (Enter)",
        command=start_duration,
        font=("Meiryo", 16, "bold"),
        width=24,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    tk.Button(
        button_frame,
        text="キャンセル / 終了 (Esc)",
        command=cancel_duration,
        font=("Meiryo", 16),
        width=20,
        height=2,
        bg="white"
    ).pack(side="left", padx=15)

    root.lift()
    root.focus_force()
    root.focus_set()
    root.mainloop()

    return selection_result[0]


# ============================================================
# 開始インストラクション
# ============================================================

def show_start_instruction():
    root = tk.Tk()
    root.title("開始インストラクション")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    tk.Label(
        root,
        text=(
            "【復唱課題（Repetition）】\n\n"
            "音声を聞いた後、画面に文字は表示されません。\n"
            "音声の通りに声に出して復唱し、終わったら\n"
            "Enter キーを押してください。\n\n"
            "準備ができたら Enter キーを押してください。（Esc キーで終了）"
        ),
        font=("Meiryo", 32, "bold"),
        bg="lightblue",
        justify="center"
    ).pack(expand=True)

    started = [False]

    def start_task(event=None):
        if started[0]:
            return
        started[0] = True
        try:
            root.destroy()
        except Exception:
            pass

    def cancel_task(event=None):
        started[0] = False
        try:
            root.destroy()
        except Exception:
            pass

    root.bind_all("<Return>", start_task)
    root.bind_all("<Escape>", cancel_task)

    root.lift()
    root.focus_force()
    root.focus_set()
    root.mainloop()

    time.sleep(0.1)
    return started[0]


# ============================================================
# Whisper 音声認識
# ============================================================

def transcribe_audio(audio_path, model):
    if not os.path.exists(audio_path):
        return {"text": "", "utterances": []}

    try:
        audio_data, _ = sf.read(audio_path, dtype="float32")
        if len(audio_data) == 0:
            return {"text": "", "utterances": []}
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        if rms < SILENCE_RMS_THRESHOLD:
            print(f"\n[情報] 録音音声がほぼ無音（RMS: {rms:.5f} < 閾値: {SILENCE_RMS_THRESHOLD}）のため、文字起こしをスキップします。")
            return {"text": "", "utterances": []}
    except Exception as e:
        print(f"[警告] 音量チェックエラー: {e}")

    print("\n文字起こしを実行中...")
    try:
        result = model.transcribe(
            audio_path,
            language="ja",
            task="transcribe",
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            no_speech_threshold=0.4
        )
    except Exception as e:
        print(f"[エラー] Whisperの文字起こしに失敗しました: {e}")
        return {"text": "", "utterances": []}

    utterances = [
        segment["text"].strip()
        for segment in result.get("segments", [])
        if segment.get("text", "").strip()
    ]
    full_text = " ".join(utterances).strip()

    for phrase in [
        "ご視聴ありがとうございました",
        "ありがとうございました",
        "ご視聴",
        "Thank you for watching",
        "Thanks for watching"
    ]:
        full_text = full_text.replace(phrase, "")

    return {
        "text": full_text.strip(),
        "utterances": utterances
    }


# ============================================================
# テキスト評価・正誤判定 ＆ モーラ正答率算出関数
# ============================================================

def evaluate_single_text(stimulus_text, transcription_text):
    clean_transcription = transcription_text.replace(" ", "").replace("。", "").replace("、", "")
    if not clean_transcription or "（無音" in clean_transcription:
        return "不正解", 0.0

    norm_stimulus = normalize_text_for_evaluation(stimulus_text)
    norm_transcription = normalize_text_for_evaluation(transcription_text)

    if not norm_stimulus:
        return "不正解", 0.0

    is_correct = (norm_stimulus == norm_transcription)
    judgment = "正解" if is_correct else "不正解"

    target_moras = get_mora_list(norm_stimulus)
    trans_moras = get_mora_list(norm_transcription)
    if target_moras:
        mora_matcher = difflib.SequenceMatcher(None, target_moras, trans_moras)
        mora_match_len = sum(triple[2] for triple in mora_matcher.get_matching_blocks())
        mora_accuracy = min(100.0, round((mora_match_len / len(target_moras)) * 100.0, 1))
    else:
        mora_accuracy = 0.0

    return judgment, mora_accuracy


# ============================================================
# 試行GUI処理
# ============================================================

def run_trial_gui(
    stimulus_text,
    stimulus_audio_path,
    output_filename,
    whisper_model,
    enable_hints=True,
    max_duration=15,
    fs=FS
):
    root = tk.Tk()
    root.title("復唱課題")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="white")

    tk.Label(
        root,
        text="【復唱課題】",
        font=("Meiryo", 36, "bold"),
        bg="white",
        fg="darkblue"
    ).pack(pady=(150, 40))

    status_label = tk.Label(
        root,
        text="音声を聴いています...",
        font=("Meiryo", 24, "bold"),
        fg="white",
        bg="gray",
        pady=25,
        justify="center"
    )
    status_label.pack(fill=tk.X, side=tk.BOTTOM)

    root.update_idletasks()
    root.update()

    # 1. 刺激音声の再生関数
    def play_stimulus_audio():
        if stimulus_audio_path and os.path.exists(stimulus_audio_path):
            try:
                data, file_fs = sf.read(stimulus_audio_path, dtype="float32")
                sd.play(data, file_fs)
                sd.wait()
            except Exception as e:
                print(f"[警告] 刺激音声の再生に失敗しました: {e}")
        else:
            speak_text_tts(stimulus_text)

    play_stimulus_audio()
    time.sleep(0.3)

    # 2. 録音の準備と開始
    status_label.config(
        text=f"【録音中】聞いた音声を復唱してください。終わったら Enter キーを押してください（{max_duration}秒制限 / Escで終了）。",
        fg="white",
        bg="red"
    )
    root.update_idletasks()
    root.update()

    audio_buffer_1 = []
    recording_active_1 = threading.Event()
    stop_recording_1 = threading.Event()
    stream_ready_1 = threading.Event()

    text_shown_time = [time.perf_counter()]
    first_speech_time = [None]
    timeout_timer_1 = [None]
    is_timeout_1 = [False]
    trial_result_container = {}

    def audio_monitor_1():
        def callback(indata, frames, time_info, status):
            if recording_active_1.is_set():
                audio_buffer_1.append(indata.copy())
                if first_speech_time[0] is None and text_shown_time[0] is not None:
                    if np.max(np.abs(indata)) > VOICE_THRESHOLD:
                        first_speech_time[0] = time.perf_counter()

        try:
            with sd.InputStream(samplerate=fs, channels=1, callback=callback, blocksize=512, dtype="float32"):
                stream_ready_1.set()
                while not stop_recording_1.is_set():
                    time.sleep(0.005)
        except Exception:
            stream_ready_1.set()

    thread_1 = threading.Thread(target=audio_monitor_1, daemon=True)
    thread_1.start()
    stream_ready_1.wait(timeout=3)
    recording_active_1.set()

    def cancel_all(event=None):
        stop_recording_1.set()
        if timeout_timer_1[0] is not None:
            try:
                root.after_cancel(timeout_timer_1[0])
            except Exception:
                pass
        try:
            root.quit()
        except Exception:
            pass

    def finish_trial_1(event=None, timed_out=False):
        if stop_recording_1.is_set():
            return
        stop_recording_1.set()
        is_timeout_1[0] = timed_out

        if timeout_timer_1[0] is not None:
            try:
                root.after_cancel(timeout_timer_1[0])
            except Exception:
                pass

        status_label.config(
            text="【解析中】音声を評価しています。しばらくお待ちください...",
            fg="white",
            bg="gray"
        )
        root.update_idletasks()
        root.update()

        if timed_out:
            t1_duration = ""
        else:
            if first_speech_time[0] is not None and text_shown_time[0] is not None:
                t1_duration = round(first_speech_time[0] - text_shown_time[0], 3)
            else:
                t1_duration = 0.0

        if not audio_buffer_1:
            root.quit()
            return

        rec_audio_1 = np.concatenate(audio_buffer_1, axis=0)
        temp_file_1 = output_filename.replace(".wav", "_temp1.wav")
        sf.write(temp_file_1, rec_audio_1, fs)

        trans_res_1 = transcribe_audio(temp_file_1, whisper_model)
        t1_text = trans_res_1.get("text", "")

        if timed_out or not t1_text:
            t1_judgment = "タイムアウト" if timed_out else "不正解"
            _, t1_mora_acc = evaluate_single_text(stimulus_text, "")
            if not t1_text:
                t1_text = "（無音または認識結果なし）"
        else:
            t1_judgment, t1_mora_acc = evaluate_single_text(stimulus_text, t1_text)

        try:
            if os.path.exists(temp_file_1):
                os.remove(temp_file_1)
        except Exception:
            pass

        if t1_judgment == "正解" or not enable_hints:
            sf.write(output_filename, rec_audio_1, fs)

            rec_duration_sec = round(len(rec_audio_1) / fs, 3)
            rec_duration_min = round(rec_duration_sec / 60.0, 4)
            mora_count = count_moras(stimulus_text)
            wpm = round((mora_count / rec_duration_sec) * 60.0, 2) if rec_duration_sec > 0 else 0.0
            wps = round(mora_count / rec_duration_sec, 2) if rec_duration_sec > 0 else 0.0

            spec_image_path = output_filename.replace(".wav", "_spectrogram.png")
            high_freq_ratio, spec_filename = analyze_and_plot_formants(output_filename, spec_image_path)

            display_t1_judgment = "タイムアウト" if is_timeout_1[0] else t1_judgment
            final_eval = "正解" if t1_judgment == "正解" else ("タイムアウト" if is_timeout_1[0] else "不正解")
            final_status = "一発正解" if t1_judgment == "正解" else ("一発タイムアウト" if is_timeout_1[0] else "一発不正解")
            if not enable_hints:
                final_status += "（ヒントなしモード）"

            trial_result_container["metrics"] = {
                "1回目認識結果": t1_text,
                "1回目正誤": display_t1_judgment,
                "1回目モーラ正答率(%)": t1_mora_acc,
                "1回目音声入力時間(秒)": t1_duration,
                "再提示有無": "なし",
                "2回目認識結果": "",
                "2回目正誤": "",
                "2回目モーラ正答率(%)": "",
                "2回目音声入力時間(秒)": "",
                "最終判定": final_eval,
                "試行ステータス": final_status,
                "統合録音ファイル": os.path.basename(output_filename),
                "高周波成分比率(%)": high_freq_ratio,
                "スペクトログラム画像": spec_filename,
                "録音時間_s": rec_duration_sec,
                "録音時間_min": rec_duration_min,
                "発話速度_moras_per_min": wpm,
                "発話速度_moras_per_sec": wps,
                "最大制限時間_s": max_duration
            }
            root.quit()
        else:
            status_label.config(
                text=f"【誤り】刺激音声を再再生中。もう一度復唱してください（{max_duration}秒制限）。",
                fg="white",
                bg="orange"
            )
            root.update_idletasks()
            root.update()

            audio_buffer_2 = []
            recording_active_2 = threading.Event()
            stop_recording_2 = threading.Event()
            t2_start_time = [None]
            timeout_timer_2 = [None]
            is_timeout_2 = [False]

            def audio_monitor_2():
                def cb2(indata, frames, time_info, status):
                    if recording_active_2.is_set():
                        audio_buffer_2.append(indata.copy())
                        if t2_start_time[0] is None:
                            if np.max(np.abs(indata)) > VOICE_THRESHOLD:
                                t2_start_time[0] = time.perf_counter()
                try:
                    with sd.InputStream(samplerate=fs, channels=1, callback=cb2, blocksize=512, dtype="float32"):
                        while not stop_recording_2.is_set():
                            time.sleep(0.005)
                except Exception:
                    pass

            thread_2 = threading.Thread(target=audio_monitor_2, daemon=True)
            thread_2.start()

            play_stimulus_audio()

            t2_prompt_start = time.perf_counter()
            recording_active_2.set()

            status_label.config(
                text=f"【再録音中】音声再提示後、もう一度復唱してください（{max_duration}秒制限 / Escで終了）。",
                fg="white",
                bg="red"
            )
            root.update_idletasks()
            root.update()

            def finish_trial_2(event=None, timed_out=False):
                if stop_recording_2.is_set():
                    return
                stop_recording_2.set()
                is_timeout_2[0] = timed_out

                if timeout_timer_2[0] is not None:
                    try:
                        root.after_cancel(timeout_timer_2[0])
                    except Exception:
                        pass
                time.sleep(0.05)

                t2_end_time = time.perf_counter()
                if timed_out:
                    t2_duration = ""
                else:
                    if t2_start_time[0] is not None:
                        t2_duration = round(t2_end_time - t2_start_time[0], 3)
                    else:
                        t2_duration = round(t2_end_time - t2_prompt_start, 3)

                rec_audio_2 = np.concatenate(audio_buffer_2, axis=0) if audio_buffer_2 else np.zeros((0, 1), dtype=np.float32)

                temp_file_2 = output_filename.replace(".wav", "_temp2.wav")
                sf.write(temp_file_2, rec_audio_2, fs)

                trans_res_2 = transcribe_audio(temp_file_2, whisper_model)
                t2_text = trans_res_2.get("text", "")

                if timed_out:
                    t2_judgment = "タイムアウト"
                    _, t2_mora_acc = evaluate_single_text(stimulus_text, "")
                    if not t2_text:
                        t2_text = "（無音または認識結果なし）"
                elif not t2_text:
                    t2_text = "（無音または認識結果なし）"
                    t2_judgment = "不正解"
                    _, t2_mora_acc = evaluate_single_text(stimulus_text, "")
                else:
                    t2_judgment, t2_mora_acc = evaluate_single_text(stimulus_text, t2_text)

                try:
                    if os.path.exists(temp_file_2):
                        os.remove(temp_file_2)
                except Exception:
                    pass

                click_raw = generate_click_sound(fs)
                click_arr = click_raw[:, np.newaxis] if click_raw.ndim == 1 else click_raw
                combined_trial_audio = np.concatenate([rec_audio_1, click_arr, rec_audio_2], axis=0)
                sf.write(output_filename, combined_trial_audio, fs)

                rec_duration_sec = round(len(rec_audio_1) / fs, 3)
                rec_duration_min = round(rec_duration_sec / 60.0, 4)
                mora_count = count_moras(stimulus_text)
                wpm = round((mora_count / rec_duration_sec) * 60.0, 2) if rec_duration_sec > 0 else 0.0
                wps = round(mora_count / rec_duration_sec, 2) if rec_duration_sec > 0 else 0.0

                spec_image_path = output_filename.replace(".wav", "_spectrogram.png")
                high_freq_ratio, spec_filename = analyze_and_plot_formants(output_filename, spec_image_path)

                if t2_judgment == "正解":
                    final_status = "ヒント後正解"
                    final_eval = "正解"
                elif t2_judgment == "タイムアウト":
                    final_status = "ヒント後タイムアウト"
                    final_eval = "タイムアウト"
                else:
                    final_status = "ヒント後不正解"
                    final_eval = "不正解"

                display_t1_judgment = "タイムアウト" if is_timeout_1[0] else t1_judgment

                trial_result_container["metrics"] = {
                    "1回目認識結果": t1_text,
                    "1回目正誤": display_t1_judgment,
                    "1回目モーラ正答率(%)": t1_mora_acc,
                    "1回目音声入力時間(秒)": t1_duration,
                    "再提示有無": "あり",
                    "2回目認識結果": t2_text,
                    "2回目正誤": t2_judgment,
                    "2回目モーラ正答率(%)": t2_mora_acc,
                    "2回目音声入力時間(秒)": t2_duration,
                    "最終判定": final_eval,
                    "試行ステータス": final_status,
                    "統合録音ファイル": os.path.basename(output_filename),
                    "高周波成分比率(%)": high_freq_ratio,
                    "スペクトログラム画像": spec_filename,
                    "録音時間_s": rec_duration_sec,
                    "録音時間_min": rec_duration_min,
                    "発話速度_moras_per_min": wpm,
                    "発話速度_moras_per_sec": wps,
                    "最大制限時間_s": max_duration
                }
                root.quit()

            def on_timeout_2():
                finish_trial_2(timed_out=True)

            def cancel_all_2(event=None):
                stop_recording_2.set()
                if timeout_timer_2[0] is not None:
                    try:
                        root.after_cancel(timeout_timer_2[0])
                    except Exception:
                        pass
                cancel_all()

            timeout_timer_2[0] = root.after(int(max_duration * 1000), on_timeout_2)
            root.bind("<Return>", lambda e: finish_trial_2(timed_out=False))
            root.bind("<Escape>", cancel_all_2)

    def on_timeout_1():
        finish_trial_1(timed_out=True)

    timeout_timer_1[0] = root.after(int(max_duration * 1000), on_timeout_1)
    root.bind("<Return>", lambda e: finish_trial_1(timed_out=False))
    root.bind("<Escape>", cancel_all)

    root.lift()
    root.focus_force()
    root.focus_set()
    root.mainloop()

    try:
        root.destroy()
    except Exception:
        pass

    return trial_result_container.get("metrics", None)


# ============================================================
# メイン実行ブロック
# ============================================================
if __name__ == "__main__":
    print(f"起動引数: {sys.argv}")

    if len(sys.argv) >= 3:
        participant_id = re.sub(r'[\\/*?:"<>|]', "", sys.argv[1].strip())
        participant_name = re.sub(r'[\\/*?:"<>|]', "", sys.argv[2].strip())
    else:
        p_info = input_participant_info()
        if not p_info:
            print("被験者情報の入力がキャンセルされました。終了します。")
            sys.exit(0)
        participant_id, participant_name = p_info

    model_name = select_model()
    if not model_name:
        print("モデル選択がキャンセルされました。終了します。")
        sys.exit(0)

    print(f"選択されたモデル [{model_name}] をロードしています...")
    whisper_model = whisper.load_model(model_name)

    enable_hints = select_experiment_mode()
    if enable_hints is None:
        print("終了します。")
        sys.exit(0)

    max_duration = select_duration_setting()
    if not max_duration:
        print("終了します。")
        sys.exit(0)

    selected_source = select_stimulus_source_gui()
    if not selected_source:
        print("刺激データの選択がキャンセルされました。終了します。")
        sys.exit(0)

    if selected_source["type"] == "wav":
        all_loaded_stimuli = load_stimuli_from_audio_folder(selected_source["path"])
    else:
        all_loaded_stimuli = load_stimuli_from_txt(selected_source["path"])

    if not all_loaded_stimuli:
        messagebox.showerror("データエラー", "選択されたソースから有効な刺激データを読み込めませんでした。")
        sys.exit(0)

    print(f"読み込んだ刺激候補の数: {len(all_loaded_stimuli)}件")

    stimuli_pairs = select_stimuli_gui(all_loaded_stimuli, source_type=selected_source["type"])
    if stimuli_pairs is None or len(stimuli_pairs) == 0:
        print("刺激の選択がキャンセルされたため、終了します。")
        sys.exit(0)
    print(f"選択された実施刺激の数: {len(stimuli_pairs)}件")

    if not show_start_instruction():
        print("課題がキャンセルされました。終了します。")
        sys.exit(0)

    timestamp_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hint_tag = "hintON" if enable_hints else "hintOFF"
    subject_dir_name = f"{participant_id}_{participant_name}"
    
    SUBJECT_DIR = os.path.join(RESULTS_DIR, subject_dir_name)
    os.makedirs(SUBJECT_DIR, exist_ok=True)

    # 1. 日付フォルダ（YYYYMMDD形式）を作成
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    DATE_DIR = os.path.join(SUBJECT_DIR, date_str)
    os.makedirs(DATE_DIR, exist_ok=True)

    # 2. 日付フォルダの下に「課題名_ヒント有無_日時_ID_名前」フォルダを作成
    session_dir_name = f"{TASK_NAME}_{hint_tag}_{timestamp_run}_{participant_id}_{participant_name}"
    session_output_dir = os.path.join(DATE_DIR, session_dir_name)
    os.makedirs(session_output_dir, exist_ok=True)

    print(f"結果保存先フォルダ: {session_output_dir}")

    csv_filename = os.path.join(session_output_dir, f"{session_dir_name}.csv")

    csv_headers = [
        "試行番号", "刺激文", "1回目認識結果", "1回目正誤", "1回目モーラ正答率(%)",
        "1回目音声入力時間(秒)", "再提示有無", "2回目認識結果", "2回目正誤", "2回目モーラ正答率(%)",
        "2回目音声入力時間(秒)", "最終判定", "試行ステータス", "統合録音ファイル",
        "高周波成分比率(%)", "スペクトログラム画像",
        "録音時間_s", "録音時間_min", "発話速度_moras_per_min",
        "発話速度_moras_per_sec", "最大制限時間_s"
    ]

    with open(csv_filename, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

        for idx, (stimulus_text, stimulus_audio_path) in enumerate(stimuli_pairs, start=1):
            output_wav = os.path.join(session_output_dir, f"trial_{idx:03d}_{timestamp_run}.wav")

            audio_info = os.path.basename(stimulus_audio_path) if stimulus_audio_path else "機械音声(TTS)"
            print(f"\n--- 試行 {idx}/{len(stimuli_pairs)}: [{stimulus_text}] (提示方法: {audio_info}) ---")
            result_metrics = run_trial_gui(
                stimulus_text=stimulus_text,
                stimulus_audio_path=stimulus_audio_path,
                output_filename=output_wav,
                whisper_model=whisper_model,
                enable_hints=enable_hints,
                max_duration=max_duration
            )

            if result_metrics:
                row_data = {"試行番号": idx, "刺激文": stimulus_text}
                row_data.update(result_metrics)
                writer.writerow(row_data)
                f.flush()
                print(
                    f"判定: {result_metrics.get('最終判定')} | "
                    f"モーラ正答率: {result_metrics.get('1回目モーラ正答率(%)')}%"
                )
            else:
                print("Escキーまたは中断操作により、課題を終了しました。")
                break

    print(f"\n=== すべての復唱課題が終了しました ===")
    print(f"保存先フォルダ: {session_output_dir}")