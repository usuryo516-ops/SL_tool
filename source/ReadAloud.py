# ============================================================
# ReadAloud.py (F1〜F3フォルマント重畳・モーラベース発話速度対応版)
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


# ============================================================
# 基本設定
# ============================================================

FS = 44100
DEFAULT_WHISPER_MODEL_NAME = "turbo"
DEFAULT_MAX_DURATION = 15  # デフォルト制限時間：15秒[cite: 4]

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

TEXT_FOLDER = os.path.join(SL_TOOL_DIR, "stimuli", "ReadAloud")
RESULTS_DIR = os.path.join(SL_TOOL_DIR, "results")
MODEL_DIR = os.path.join(SL_TOOL_DIR, "whisper_model")

os.environ["PATH"] = SL_TOOL_DIR + os.pathsep + os.environ.get("PATH", "")


# ============================================================
# モーラ数（拍数）の計算関数
# ============================================================
def count_moras(text):
    """
    提示テキストからJanomeを使って読み（カタカナ）を取得し、
    正確なモーラ数（拍数）を計算する関数
    """
    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(text))
    
    kata_to_hira = str.maketrans(
        'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォッャュョー',
        'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふめほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだヂヅデドバビブベボパピプペポァィゥェォッャュョー'
    )

    full_reading = ""
    for token in tokens:
        if hasattr(token, 'reading') and token.reading and token.reading != '*':
            full_reading += token.reading
        else:
            full_reading += token.surface

    hira_text = full_reading.translate(kata_to_hira)
    mora_count = 0
    ignore_chars = set("ぁぃぅぇぉゃゅょ")
    
    for char in hira_text:
        if char in ignore_chars:
            continue
        mora_count += 1

    return max(1, mora_count)


# ============================================================
# 構音の歪み評価 ＆ フォルマント付きスペクトログラム生成関数
# ============================================================
def analyze_and_plot_formants(audio_path, output_image_path):
    """
    フィルター処理を一切行せず、抽出されたF1〜F3の生データを
    そのままスペクトログラムにプロットする関数。
    """
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
# スマートヒント文字列の抽出
# ============================================================

def get_hint_text(stimulus_text):
    if "|" in stimulus_text:
        parts = stimulus_text.split("|")
        return parts[0].strip()

    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(stimulus_text))
    
    if " " not in stimulus_text and "。" not in stimulus_text and "、" not in stimulus_text:
        if tokens and hasattr(tokens[0], 'reading') and tokens[0].reading and tokens[0].reading != '*':
            reading_kata = tokens[0].reading
            reading_hira = reading_kata.translate(str.maketrans(
                'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォッャュョー',
                'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふめほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだヂヅデドバビブベボパピプペポァィゥェォッャュョー'
            ))
            return reading_hira[0]
        else:
            return stimulus_text[0]
    else:
        if len(tokens) >= 2:
            return tokens[0].surface + tokens[1].surface
        elif tokens:
            return tokens[0].surface
        else:
            return stimulus_text[:2]


# ============================================================
# Whisper モデル選択 GUI
# ============================================================

def select_model():
    root = tk.Tk()
    root.title("Whisper モデル選択")
    root.configure(bg="lightgreen")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    selected_model = tk.StringVar(value=DEFAULT_WHISPER_MODEL_NAME)
    selection_result = [None]

    container = tk.Frame(root, bg="lightgreen")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【Whisper 音声認識モデルの選択】",
        font=("Meiryo", 32, "bold"),
        bg="lightgreen",
        fg="black"
    ).pack(pady=(0, 15))

    tk.Label(
        container,
        text="文字起こしに使用するモデルを選択してください。\n（上下キーで選択変更、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightgreen",
        fg="darkgreen",
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

    button_frame = tk.Frame(container, bg="lightgreen")
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
# 実験モード選択 GUI（ヒントあり / なし）
# ============================================================

def select_experiment_mode():
    root = tk.Tk()
    root.title("実験モード選択")
    root.configure(bg="lightgreen")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    selected_mode = tk.BooleanVar(value=True)
    selection_result = [None]

    container = tk.Frame(root, bg="lightgreen")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【実験モードの選択】",
        font=("Meiryo", 32, "bold"),
        bg="lightgreen",
        fg="black"
    ).pack(pady=(0, 15))

    tk.Label(
        container,
        text="ヒント機能の有無を選択してください。\n（上下キーで選択変更、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightgreen",
        fg="darkgreen",
        justify="center"
    ).pack(pady=(0, 30))

    mode_frame = tk.Frame(container, bg="white", padx=40, pady=30)
    mode_frame.pack(pady=10)

    options = [
        (True, "ヒントあり", "誤答時にヒントを音声再生し、再録音を実施します（標準）"),
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

    button_frame = tk.Frame(container, bg="lightgreen")
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
    root.configure(bg="lightgreen")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    duration_var = tk.StringVar(value=str(DEFAULT_MAX_DURATION))
    selection_result = [None]

    container = tk.Frame(root, bg="lightgreen")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【課題制限時間の設定】",
        font=("Meiryo", 32, "bold"),
        bg="lightgreen",
        fg="black"
    ).pack(pady=(0, 15))

    tk.Label(
        container,
        text="1試行あたりの制限時間（秒）を半角数字で入力してください。\n（指定時間を超えると自動的に録音が終了します）",
        font=("Meiryo", 16),
        bg="lightgreen",
        fg="darkgreen",
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

    button_frame = tk.Frame(container, bg="lightgreen")
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
    root.configure(bg="lightgreen")

    tk.Label(
        root,
        text=(
            "【音読課題】\n\n"
            "画面に表示される文章を声に出して読み、\n"
            "終わったら Enter キーを押してください。\n\n"
            "準備ができたら Enter キーを押してください。"
        ),
        font=("Meiryo", 32, "bold"),
        bg="lightgreen",
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
    root.bind_all("<space>", start_task)
    root.bind_all("<Escape>", cancel_task)

    root.lift()
    root.focus_force()
    root.focus_set()
    root.mainloop()

    time.sleep(0.1)
    return started[0]


# ============================================================
# Whisper 音声認識（無音RMSチェック付き）
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
# 単体のテキスト評価関数
# ============================================================

def evaluate_single_text(stimulus_text, transcription_text):
    clean_stimulus = stimulus_text.replace("|", "").replace(" ", "").replace("。", "").replace("、", "")
    clean_transcription = transcription_text.replace(" ", "").replace("。", "").replace("、", "")
    if not clean_transcription or "（無音" in clean_transcription:
        return "不正解"
    is_correct = (clean_stimulus == clean_transcription)
    return "正解" if is_correct else "不正解"


# ============================================================
# 試行GUI処理
# ============================================================

def run_trial_gui(
    text_content,
    output_filename,
    whisper_model,
    enable_hints=True,
    max_duration=15,
    fs=FS
):
    display_text = text_content.replace("|", "")

    exit_requested = [False]
    audio_buffer_1 = []

    recording_active_1 = threading.Event()
    stop_recording_1 = threading.Event()
    stream_ready_1 = threading.Event()

    text_shown_time = [None]
    first_speech_time = [None]
    timeout_timer_1 = [None]
    is_timeout_1 = [False]

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
                while text_shown_time[0] is None:
                    time.sleep(0.001)
                while not stop_recording_1.is_set():
                    time.sleep(0.005)
        except Exception:
            stream_ready_1.set()

    thread_1 = threading.Thread(target=audio_monitor_1, daemon=True)
    thread_1.start()
    stream_ready_1.wait(timeout=3)

    root = tk.Tk()
    root.title("音読課題")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="white")

    tk.Label(
        root,
        text=display_text,
        font=("Meiryo", 48, "bold"),
        bg="white",
        fg="black",
        wraplength=1200,
        justify="center"
    ).pack(expand=True, fill=tk.BOTH)

    status_label = tk.Label(
        root,
        text=f"【録音中】文章を音読してください。終了したら Enter キーを押してください（{max_duration}秒制限）。",
        font=("Meiryo", 22, "bold"),
        fg="white",
        bg="red",
        pady=18,
        justify="center"
    )
    status_label.pack(fill=tk.X, side=tk.BOTTOM)

    root.update_idletasks()
    root.update()

    text_shown_time[0] = time.perf_counter()
    recording_active_1.set()

    trial_result_container = {}

    def cancel_all(event=None):
        exit_requested[0] = True
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
            if not t1_text:
                t1_text = "（無音または認識結果なし）"
        else:
            t1_judgment = evaluate_single_text(text_content, t1_text)

        try:
            if os.path.exists(temp_file_1):
                os.remove(temp_file_1)
        except Exception:
            pass

        if t1_judgment == "正解" or not enable_hints:
            sf.write(output_filename, rec_audio_1, fs)
            
            rec_duration_sec = round(len(rec_audio_1) / fs, 3)
            rec_duration_min = round(rec_duration_sec / 60.0, 4)
            mora_count = count_moras(display_text)
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
                "1回目音声入力時間(秒)": t1_duration,
                "再提示有無": "なし",
                "2回目認識結果": "",
                "2回目正誤": "",
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
            hint_str = get_hint_text(text_content)
            status_label.config(
                text=f"【誤り】ヒントを再生中。続けてもう一度お読みください（{max_duration}秒制限）。",
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

            t2_prompt_start = time.perf_counter()
            recording_active_2.set()

            def speak_hint_concurrently():
                try:
                    escaped = hint_str.replace("'", "''")
                    cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Volume = 100; $synth.Speak(\'{escaped}\')"'
                    subprocess.run(cmd, shell=True)
                except Exception:
                    pass
            threading.Thread(target=speak_hint_concurrently, daemon=True).start()

            status_label.config(
                text=f"【再録音中】もう一度音読してください。終了したら Enter キーを押してください（{max_duration}秒制限）。",
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
                    if not t2_text:
                        t2_text = "（無音または認識結果なし）"
                elif not t2_text:
                    t2_text = "（無音または認識結果なし）"
                    t2_judgment = "不正解"
                else:
                    t2_judgment = evaluate_single_text(text_content, t2_text)

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
                mora_count = count_moras(display_text)
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
                    "1回目音声入力時間(秒)": t1_duration,
                    "再提示有無": "あり",
                    "2回目認識結果": t2_text,
                    "2回目正誤": t2_judgment,
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

            for k in ("<Return>", "<Escape>", "<space>"):
                root.unbind_all(k)
            root.bind_all("<Return>", lambda e: finish_trial_2(e, timed_out=False))
            root.bind_all("<space>", lambda e: finish_trial_2(e, timed_out=False))
            root.bind_all("<Escape>", cancel_all)

            timeout_timer_2[0] = root.after(int(max_duration * 1000), lambda: finish_trial_2(timed_out=True))

    timeout_timer_1[0] = root.after(int(max_duration * 1000), lambda: finish_trial_1(timed_out=True))

    root.bind_all("<Return>", lambda e: finish_trial_1(e, timed_out=False))
    root.bind_all("<space>", lambda e: finish_trial_1(e, timed_out=False))
    root.bind_all("<Escape>", cancel_all)

    def keep_focus():
        try:
            if root.winfo_exists() and not stop_recording_1.is_set():
                root.lift()
                root.focus_force()
                root.focus_set()
                root.after(250, keep_focus)
        except Exception:
            pass

    root.lift()
    root.focus_force()
    root.focus_set()
    root.after(250, keep_focus)
    root.mainloop()

    try:
        root.destroy()
    except Exception:
        pass

    stop_recording_1.set()
    thread_1.join()

    metrics = trial_result_container.get("metrics", None)
    return output_filename, metrics, exit_requested[0]


# ============================================================
# テキスト刺激読み込み
# ============================================================

def load_read_aloud_stimuli(folder_path):
    stimuli = []
    if not os.path.exists(folder_path):
        return stimuli

    txt_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".txt")
    ])

    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_text = line.strip()
                    if line_text:
                        clean_text = re.sub(r"^\d+_", "", line_text)
                        stimuli.append({
                            "source_file": filename,
                            "text": clean_text
                        })
        except Exception as e:
            print(f"[エラー] ファイルの読み込みに失敗しました ({filename}): {e}")

    return stimuli


# ============================================================
# 統合 CSV 保存
# ============================================================

def save_total_results_csv(
    all_results,
    output_filename,
    whisper_model_name
):
    headers = [
        "被験者ID",
        "試行番号",
        "提示文",
        "使用モデル",
        "最大制限時間_s",
        "1回目認識結果",
        "1回目正誤",
        "1回目音声入力時間(秒)",
        "再提示有無",
        "2回目認識結果",
        "2回目正誤",
        "2回目音声入力時間(秒)",
        "最終判定",
        "試行ステータス",
        "統合録音ファイル",
        "高周波成分比率(%)",
        "スペクトログラム画像",
        "録音時間_s",
        "録音時間_min",
        "発話速度_moras_per_min",
        "発話速度_moras_per_sec"
    ]

    with open(output_filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        for res in all_results:
            m = res.get("metrics", {})
            writer.writerow([
                res.get("subject_id", ""),
                res.get("trial_num", ""),
                res.get("stimulus_text", ""),
                whisper_model_name,
                m.get("最大制限時間_s", 15),
                m.get("1回目認識結果", ""),
                m.get("1回目正誤", ""),
                m.get("1回目音声入力時間(秒)", ""),
                m.get("再提示有無", ""),
                m.get("2回目認識結果", ""),
                m.get("2回目正誤", ""),
                m.get("2回目音声入力時間(秒)", ""),
                m.get("最終判定", ""),
                m.get("試行ステータス", ""),
                m.get("統合録音ファイル", ""),
                m.get("高周波成分比率(%)", 0.0),
                m.get("スペクトログラム画像", ""),
                m.get("録音時間_s", 0.0),
                m.get("録音時間_min", 0.0),
                m.get("発話速度_moras_per_min", 0.0),
                m.get("発話速度_moras_per_sec", 0.0)
            ])


# ============================================================
# メイン実行処理
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python ReadAloud.py [被験者ID] [被験者名(省略可)]")
        sys.exit(1)

    subject_id = sys.argv[1]
    subject_name = sys.argv[2] if len(sys.argv) >= 3 else ""

    # 1. モデル選択 GUI
    selected_model_name = select_model()
    if not selected_model_name:
        print("モデル選択がキャンセルされました。終了します。")
        sys.exit(0)

    # 2. 実験モード選択 GUI（ヒントあり／なし）
    enable_hints = select_experiment_mode()
    if enable_hints is None:
        print("実験モード選択がキャンセルされました。終了します。")
        sys.exit(0)

    # 3. 制限時間設定 GUI
    max_duration = select_duration_setting()
    if max_duration is None:
        print("制限時間設定がキャンセルされました。終了します。")
        sys.exit(0)

    # 4. Whisper モデルのロード
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\nWhisperモデル ({selected_model_name}) をロード中...")
    model = whisper.load_model(selected_model_name, download_root=MODEL_DIR)

    # 5. 開始インストラクション表示
    if not show_start_instruction():
        print("開始インストラクションでキャンセルされました。終了します。")
        sys.exit(0)

    # 6. 刺激テキストの読み込み
    stimuli = load_read_aloud_stimuli(TEXT_FOLDER)
    if not stimuli:
        print(f"[エラー] 刺激テキストが見つかりません: {TEXT_FOLDER}")
        sys.exit(1)

    # 7. 出力フォルダ・ファイル名のプレフィックス準備
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{subject_id}_{subject_name}" if subject_name else subject_id
    hint_tag = "hintON" if enable_hints else "hintOFF"
    
    output_subfolder = os.path.join(RESULTS_DIR, folder_name, f"ReadAloud_{hint_tag}_{timestamp}_{folder_name}")
    os.makedirs(output_subfolder, exist_ok=True)

    all_trial_results = []

    # 8. 試行ループ
    for idx, item in enumerate(stimuli, start=1):
        stim_text = item["text"]
        print(f"\n--- 試行 {idx}/{len(stimuli)}: {stim_text} ---")

        out_wav = os.path.join(output_subfolder, f"ReadAloud_{timestamp}_{folder_name}_trial_{idx:02d}.wav")
        saved_wav, metrics, exit_requested = run_trial_gui(
            text_content=stim_text,
            output_filename=out_wav,
            whisper_model=model,
            enable_hints=enable_hints,
            max_duration=max_duration,
            fs=FS
        )

        if exit_requested:
            print("\n[中断] ユーザー操作により実験が中断されました。")
            break

        if metrics:
            all_trial_results.append({
                "subject_id": subject_id,
                "trial_num": idx,
                "stimulus_text": stim_text,
                "metrics": metrics
            })

    # 9. 統合CSVの書き出し
    if all_trial_results:
        csv_filename = os.path.join(output_subfolder, f"ReadAloud_{timestamp}_{folder_name}_results.csv")
        save_total_results_csv(all_trial_results, csv_filename, selected_model_name)
        print(f"\n実験完了。結果CSVを保存しました: {csv_filename}")
    else:
        print("\n記録された結果はありません。")