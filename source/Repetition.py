# ============================================================
# Repetition.py (ヒント選択・ファイル命名規則対応版)
# ============================================================

import csv
import datetime
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
import ctypes

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

from janome.tokenizer import Tokenizer


# ============================================================
# 基本設定
# ============================================================

FS = 44100
DEFAULT_MAX_RESPONSE_DURATION = 15  # デフォルト制限時間：15秒[cite: 5]
DEFAULT_WHISPER_MODEL_NAME = "turbo"

# RMS がこの値より低い録音は無反応・ほぼ無音として扱う
SILENCE_RMS_THRESHOLD = 0.002

# 発話開始検出の音量閾値
VOICE_ONSET_THRESHOLD = 0.08


# ============================================================
# パス設定
# ============================================================

def get_app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = get_app_dir()
SL_TOOL_DIR = os.path.dirname(APP_DIR)
RESULTS_DIR = os.path.join(SL_TOOL_DIR, "results")
MODEL_DIR = os.path.join(SL_TOOL_DIR, "whisper_model")
STIMULUS_DIR = os.path.join(SL_TOOL_DIR, "stimuli", "Repetition")

os.environ["PATH"] = (
    SL_TOOL_DIR
    + os.pathsep
    + os.environ.get("PATH", "")
)


# ============================================================
# ユーティリティ・GUI
# ============================================================

def force_window_focus(root):
    try:
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.update_idletasks()
        hwnd = root.winfo_id()
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        root.focus_force()
        root.focus_set()
    except Exception:
        pass


def generate_click_sound(fs=FS, duration=0.05, freq=1000):
    """録音1と録音2の間に挟むクリック音（ビープ音）を生成する"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    click = np.sin(2 * np.pi * freq * t) * np.hanning(len(t)) * 0.5
    return click.astype(np.float32)


def select_whisper_model():
    root = tk.Tk()
    root.title("Whisperモデル選択")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    selected_model = tk.StringVar(value=DEFAULT_WHISPER_MODEL_NAME)
    result = [None]

    tk.Label(root, text="音声認識モデルを選択してください", font=("Meiryo", 36), bg="lightblue", fg="black").pack(pady=(60, 25))

    tk.Radiobutton(root, text="Small\n標準モード：比較的高速で実用的な認識精度", variable=selected_model, value="small", font=("Meiryo", 24), bg="lightblue", fg="black", selectcolor="white", justify="left", anchor="w", padx=30, pady=15).pack(fill=tk.X, padx=250, pady=5)
    tk.Radiobutton(root, text="Turbo\n高速モード：Enter後の待機時間を短くしたい場合", variable=selected_model, value="turbo", font=("Meiryo", 24), bg="lightblue", fg="black", selectcolor="white", justify="left", anchor="w", padx=30, pady=15).pack(fill=tk.X, padx=250, pady=5)
    tk.Radiobutton(root, text="Large\n高精度モード：待ち時間が長くても精度を優先", variable=selected_model, value="large", font=("Meiryo", 24), bg="lightblue", fg="black", selectcolor="white", justify="left", anchor="w", padx=30, pady=15).pack(fill=tk.X, padx=250, pady=5)

    tk.Label(root, text="↑↓: モデル選択    1: Small    2: Turbo    3: Large\nEnter または開始ボタンで次へ\nEscape で終了", font=("Meiryo", 18), bg="lightblue", fg="black", justify="center").pack(pady=25)

    def start_task(event=None):
        result[0] = selected_model.get()
        try: root.destroy()
        except: pass
        return "break"

    def cancel_task(event=None):
        result[0] = None
        try: root.destroy()
        except: pass
        return "break"

    model_order = ["small", "turbo", "large"]
    def move_model_selection(direction):
        curr = selected_model.get()
        idx = model_order.index(curr) if curr in model_order else 0
        new_idx = max(0, min(idx + direction, len(model_order) - 1))
        selected_model.set(model_order[new_idx])
        return "break"

    tk.Button(root, text="次へ", font=("Meiryo", 28), bg="white", fg="black", command=start_task, padx=70, pady=12).pack(pady=15)

    root.bind("<Return>", start_task)
    root.bind("<KP_Enter>", start_task)
    root.bind("<Escape>", cancel_task)
    root.bind("<Up>", lambda e: move_model_selection(-1))
    root.bind("<Down>", lambda e: move_model_selection(1))
    root.bind("1", lambda e: selected_model.set("small"))
    root.bind("2", lambda e: selected_model.set("turbo"))
    root.bind("3", lambda e: selected_model.set("large"))

    root.update_idletasks()
    root.update()
    force_window_focus(root)

    while result[0] is None:
        try:
            if not root.winfo_exists():
                break
            root.update()
            root.update_idletasks()
        except Exception:
            break
        time.sleep(0.01)

    try: root.destroy()
    except: pass
    return result[0]


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
        text="ヒント（再提示）機能の有無を選択してください。\n（上下キーで選択変更、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightblue",
        fg="darkblue",
        justify="center"
    ).pack(pady=(0, 30))

    mode_frame = tk.Frame(container, bg="white", padx=40, pady=30)
    mode_frame.pack(pady=10)

    options = [
        (True, "ヒントあり", "誤答・タイムアウト時に音声を再提示し、再録音を実施します（標準）"),
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
        text="次へ (Enter)",
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


def select_duration_setting():
    root = tk.Tk()
    root.title("制限時間設定")
    root.configure(bg="lightblue")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    duration_var = tk.StringVar(value=str(DEFAULT_MAX_RESPONSE_DURATION))
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
        text="1試行あたりの制限時間（秒）を半角数字で入力してください。\n（指定時間を超えると自動的に録音が終了します）",
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
        text="課題を開始 (Enter)",
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


def show_instruction_and_wait(text, font_size=32):
    root = tk.Tk()
    root.title("インストラクション")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    tk.Label(root, text=text, font=("Meiryo", font_size), bg="lightblue", fg="black", justify="center").pack(expand=True)
    closed = [False]

    def close_window(event=None):
        if closed[0]: return "break"
        closed[0] = True
        try: root.destroy()
        except: pass
        return "break"

    root.bind("<Return>", close_window)
    root.bind("<KP_Enter>", close_window)
    root.bind("<space>", close_window)
    root.bind("<Escape>", close_window)
    force_window_focus(root)

    while not closed[0] and root.winfo_exists():
        try:
            root.update()
            root.update_idletasks()
        except:
            break
        time.sleep(0.01)
    try: root.destroy()
    except: pass
    time.sleep(0.1)


# ============================================================
# 刺激音声取得
# ============================================================

def get_stimulus_files(folder_path):
    files = []
    if not os.path.exists(folder_path):
        print(f"[エラー] 刺激音声フォルダーが存在しません: {folder_path}")
        return files

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".wav"):
            continue
        if "_" not in filename:
            continue

        number_part, sentence_with_extension = filename.split("_", 1)
        if not number_part.isdigit():
            continue

        sentence = sentence_with_extension[:-4].strip()
        if not sentence:
            continue

        files.append((int(number_part), filename, sentence))

    files.sort(key=lambda x: x[0])
    print(f"[刺激音声検出] {len(files)} 件の刺激音声が見つかりました。")
    return files


# ============================================================
# 単発の録音GUI実行関数
# ============================================================

def run_single_recording_gui(
    audio_to_play,
    play_fs,
    window_title="復唱課題",
    prompt_text="聞こえた文をそのまま繰り返してください。",
    listen_text="聞いてください。",
    fs=FS,
    max_duration=15
):
    audio_buffer = []
    stream_ready = threading.Event()
    recording_active = threading.Event()
    stop_recording = threading.Event()

    recording_started_time = [None]
    voice_onset_time = [None]
    voice_detected = [False]
    timed_out = [False]
    aborted = [False]
    audio_error = [None]

    stimulus_started = [False]
    response_started = [False]
    root = [None]

    def audio_monitor():
        def callback(indata, frames, time_info, status):
            if not recording_active.is_set():
                return
            audio_buffer.append(indata.copy())
            if not voice_detected[0]:
                volume = np.abs(indata).max()
                if volume > VOICE_ONSET_THRESHOLD:
                    onset_time = time.perf_counter() - recording_started_time[0]
                    voice_onset_time[0] = onset_time
                    voice_detected[0] = True

        try:
            with sd.InputStream(
                samplerate=fs,
                channels=1,
                callback=callback,
                blocksize=512,
                dtype="float32"
            ):
                stream_ready.set()
                while not stop_recording.is_set():
                    if recording_started_time[0] is not None:
                        elapsed = time.perf_counter() - recording_started_time[0]
                        if elapsed >= max_duration:
                            timed_out[0] = True
                            stop_recording.set()
                            if root[0] is not None:
                                try:
                                    root[0].after(0, root[0].quit)
                                except Exception:
                                    pass
                            break
                    time.sleep(0.005)
        except Exception as e:
            audio_error[0] = e
            stream_ready.set()

    audio_thread = threading.Thread(target=audio_monitor, daemon=True)
    audio_thread.start()
    stream_ready.wait(timeout=3)

    root[0] = tk.Tk()
    root[0].title(window_title)
    root[0].attributes("-topmost", True)
    root[0].attributes("-fullscreen", True)
    root[0].configure(bg="white")

    main_label = tk.Label(
        root[0],
        text=f"{listen_text}\n\nEnter キーを押すと、音声が再生されます。\n\n(途中でやめる場合は Escape キー)",
        font=("Meiryo", 36, "bold"),
        bg="white",
        fg="black",
        justify="center"
    )
    main_label.pack(expand=True)

    timer_label = tk.Label(root[0], text="", font=("Meiryo", 54, "bold"), bg="white", fg="red", pady=20)
    timer_label.pack(fill=tk.X, side=tk.BOTTOM)

    status_label = tk.Label(
        root[0],
        text=f"【準備中】Enter キーを押してください。 (中断: Escape)",
        font=("Meiryo", 22, "bold"),
        fg="blue",
        bg="lightyellow",
        pady=18
    )
    status_label.pack(fill=tk.X, side=tk.BOTTOM)

    timer_remaining = [max_duration]

    def update_timer():
        if stop_recording.is_set() or not response_started[0]:
            return
        if timer_remaining[0] > 0:
            timer_remaining[0] -= 1
            timer_label.config(text=f"残り時間: {timer_remaining[0]} 秒")
            if timer_remaining[0] <= 5:
                timer_label.config(fg="white", bg="orange")
            root[0].after(1000, update_timer)

    def abort_task(event=None):
        aborted[0] = True
        stop_recording.set()
        try:
            sd.stop()
        except Exception:
            pass
        try:
            status_label.config(text="【中断】課題を終了します...", fg="white", bg="darkred")
            root[0].quit()
        except Exception:
            pass

    def end_response(event=None):
        if not response_started[0] or stop_recording.is_set():
            return
        stop_recording.set()
        try:
            status_label.config(text="【終了】音声を保存しています...", fg="gray", bg="lightgray")
            root[0].quit()
        except Exception:
            pass

    def begin_response_recording():
        if aborted[0]:
            return
        response_started[0] = True
        recording_started_time[0] = time.perf_counter()
        recording_active.set()

        main_label.config(text=prompt_text)
        timer_label.config(text=f"残り時間: {max_duration} 秒", fg="red", bg="lightyellow")
        status_label.config(text=f"【録音中】復唱が終わったら Enter キーを押してください。（{max_duration}秒制限） (中断: Escape)", fg="white", bg="red")

        force_window_focus(root[0])
        root[0].after(1000, update_timer)

    def play_audio_in_thread():
        try:
            sd.play(audio_to_play, play_fs)
            while sd.get_stream().active:
                if aborted[0] or stop_recording.is_set():
                    break
                time.sleep(0.05)
            sd.stop()
        except Exception as e:
            print(f"[音声再生エラー] {e}")
        
        if not aborted[0] and not stop_recording.is_set():
            root[0].after(0, begin_response_recording)
        else:
            stop_recording.set()
            try:
                root[0].after(0, root[0].quit)
            except Exception:
                pass

    def start_playback(event=None):
        if stimulus_started[0] or aborted[0]:
            return
        stimulus_started[0] = True
        root[0].unbind("<Return>")
        root[0].unbind("<space>")

        main_label.config(text=listen_text)
        timer_label.config(text="", bg="white")
        status_label.config(text=f"【音声再生中】{listen_text} (中断: Escape)", fg="white", bg="steelblue")

        playback_thread = threading.Thread(target=play_audio_in_thread, daemon=True)
        playback_thread.start()

    root[0].bind("<Return>", start_playback)
    root[0].bind("<space>", start_playback)
    root[0].bind_all("<Return>", end_response)
    root[0].bind_all("<Escape>", abort_task)

    force_window_focus(root[0])
    root[0].mainloop()

    stop_recording.set()
    try:
        sd.stop()
    except Exception:
        pass
    try:
        root[0].destroy()
    except Exception:
        pass
    audio_thread.join()

    return None, timed_out[0], voice_onset_time[0], aborted[0], audio_buffer


# ============================================================
# Whisper 文字起こし & 評価関連
# ============================================================

def kana(s):
    """カタカナをひらがなに変換し、記号や長音の調整を行う"""
    return ''.join(chr(ord(c)-96) if 0x30a1 <= ord(c) <= 0x30f6 else c for c in s.strip()).replace('ー', '')

def transcribe_audio_file(audio_path, model):
    if not os.path.exists(audio_path):
        return ""
    try:
        audio_data, _ = sf.read(audio_path, dtype="float32")
        if len(audio_data) == 0:
            return ""
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        if rms < SILENCE_RMS_THRESHOLD:
            return ""
    except Exception:
        pass

    try:
        result = model.transcribe(
            audio_path,
            language="ja",
            task="transcribe",
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            no_speech_threshold=0.4
        )
    except Exception:
        return ""

    text = result["text"].strip()
    hallucinations = ["ご清聴ありがとうございました", "ご視聴ありがとうございました", "ありがとうございました"]
    for h in hallucinations:
        text = text.replace(h, "")
    return text.strip()


def analyze_repetition(reference_sentence, recognized_text):
    """
    提示文と認識結果を比較し、表記揺れ（漢字・ひらがな）を許容しつつ、
    文末の時制（です/でした等）や助動詞・内容語の欠損・置き換わりを厳密に判定する。
    """
    ref_clean = reference_sentence.strip().rstrip('。')
    rec_clean = recognized_text.strip().rstrip('。')

    if not rec_clean:
        return {
            "提示文": reference_sentence,
            "Whisper認識結果": "（なし）",
            "文全体の一致": "不正解",
            "提示語再現率": "0.0%"
        }

    tokenizer = Tokenizer()
    
    ref_tokens = [t for t in tokenizer.tokenize(ref_clean) if t.part_of_speech.split(',')[0] != '記号']
    rec_tokens = [t for t in tokenizer.tokenize(rec_clean) if t.part_of_speech.split(',')[0] != '記号']

    if abs(len(ref_tokens) - rec_tokens.__len__()) > 2:
        exact_match = False
    else:
        ref_features = [(kana(t.surface), t.part_of_speech) for t in ref_tokens]
        rec_features = [(kana(t.surface), t.part_of_speech) for t in rec_tokens]

        ref_kanas = [f[0] for f in ref_features]
        rec_kanas = [f[0] for f in rec_features]

        suffix_match = True
        if len(ref_tokens) > 0 and len(rec_tokens) > 0:
            last_ref = ref_tokens[-1]
            last_rec = rec_tokens[-1]
            if kana(last_ref.surface) != kana(last_rec.surface):
                suffix_match = False

        exact_match = (ref_kanas == rec_kanas) and suffix_match

    try:
        n_ref = len(ref_tokens)
        ref_kanas_only = [kana(t.surface) for t in ref_tokens]
        rec_kanas_only = [kana(t.surface) for t in rec_tokens]
        
        correct_count = sum(1 for r_k in ref_kanas_only if r_k in rec_kanas_only)
        recall_rate = (correct_count / n_ref * 100) if n_ref > 0 else 0.0
    except Exception:
        recall_rate = 100.0 if exact_match else 0.0

    return {
        "提示文": reference_sentence,
        "Whisper認識結果": recognized_text,
        "文全体の一致": "正解" if exact_match else "不正解",
        "提示語再現率": f"{recall_rate:.1f}%"
    }


# ============================================================
# メイン処理
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python Repetition.py [被験者ID] [被験者名]")
        sys.exit(1)

    SUBJECT_ID = sys.argv[1]
    SUBJECT_NAME = sys.argv[2] if len(sys.argv) >= 3 else ""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. モデル選択 GUI
    selected_model_name = select_whisper_model()
    if not selected_model_name:
        print("[中断] モデル選択がキャンセルされました。")
        sys.exit(0)

    # 2. 実験モード選択 GUI（ヒントあり／なし）
    enable_hints = select_experiment_mode()
    if enable_hints is None:
        print("[中断] 実験モード選択がキャンセルされました。")
        sys.exit(0)

    # 3. 制限時間設定 GUI
    max_response_duration = select_duration_setting()
    if max_response_duration is None:
        print("[中断] 制限時間設定がキャンセルされました。")
        sys.exit(0)

    subject_folder = f"{SUBJECT_ID}_{SUBJECT_NAME}" if SUBJECT_NAME else SUBJECT_ID
    SUBJECT_DIR = os.path.join(RESULTS_DIR, subject_folder)
    os.makedirs(SUBJECT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hint_tag = "hintON" if enable_hints else "hintOFF"
    
    folder_name = f"{SUBJECT_ID}_{SUBJECT_NAME}" if SUBJECT_NAME else SUBJECT_ID
    OUTPUT_DIR = os.path.join(SUBJECT_DIR, f"Repetition_{hint_tag}_{timestamp}_{folder_name}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_prefix = f"Repetition_{timestamp}_{folder_name}"

    stimulus_files = get_stimulus_files(STIMULUS_DIR)
    if not stimulus_files:
        print("[エラー] 刺激音声が見つかりません。")
        sys.exit(1)

    print(f"Whisper '{selected_model_name}' モデルをロード中...")
    try:
        WHISPER_MODEL = whisper.load_model(selected_model_name, download_root=MODEL_DIR)
    except Exception as e:
        print(f"[エラー] Whisper モデルロード失敗: {e}")
        sys.exit(1)

    show_instruction_and_wait(
        "【復唱課題】\n\n"
        f"使用モデル: {selected_model_name.upper()} / モード: {'ヒントあり' if enable_hints else 'ヒントなし'} / 制限時間: {max_response_duration}秒\n\n"
        "これから文が聞こえてきます。\n"
        "聞こえた文を、そのまま繰り返してください。\n\n"
        "※途中でやめる場合は Escape キーを押してください。\n\n"
        "準備ができたら Enter キーを押してください。"
    )

    all_results = []
    task_aborted = False

    for index, (trial_num, stimulus_filename, reference_sentence) in enumerate(stimulus_files, start=1):
        stimulus_path = os.path.join(STIMULUS_DIR, stimulus_filename)
        print(f"\n==================== 試行 {index}/{len(stimulus_files)} ====================")
        print(f"提示文: {reference_sentence}")

        try:
            stimulus_audio, stimulus_fs = sf.read(stimulus_path, dtype="float32")
        except Exception as e:
            print(f"[エラー] 刺激音声読込失敗: {e}")
            continue

        # --- 録音1 (1回目) ---
        _, timed_out_1, onset_1, aborted_1, audio_buffer_1 = run_single_recording_gui(
            audio_to_play=stimulus_audio,
            play_fs=stimulus_fs,
            window_title=f"復唱課題 - 試行 {index} (1回目)",
            prompt_text="聞こえた文をそのまま繰り返してください。",
            listen_text="聞いてください。",
            max_duration=max_response_duration
        )

        if aborted_1:
            print("[中断] ユーザーによって課題が中断されました。")
            task_aborted = True
            break

        if not audio_buffer_1:
            print("[エラー] 録音1のデータが取得できませんでした。スキップします。")
            continue

        try:
            audio_array_1 = np.concatenate(audio_buffer_1, axis=0)
        except Exception as e:
            print(f"[エラー] 配列結合失敗: {e}")
            continue

        audio_array_1 = np.squeeze(audio_array_1)

        temp_audio_1 = os.path.join(OUTPUT_DIR, f"temp_trial{trial_num:02d}_1.wav")
        sf.write(temp_audio_1, audio_array_1, FS)
        recognized_text_1 = transcribe_audio_file(temp_audio_1, WHISPER_MODEL)
        analysis_1 = analyze_repetition(reference_sentence, recognized_text_1)

        is_correct_1 = (analysis_1["文全体の一致"] == "正解")
        used_hint = False
        trial_status = ""
        
        final_audio_array = audio_array_1
        recognized_text_2 = ""
        analysis_2 = {}
        onset_2 = None

        # --- 判定：ヒント有効かつ、誤りまたはTimeoutの場合に再提示＋録音2へ ---
        if enable_hints and (not is_correct_1 or timed_out_1):
            print("[判定] 1回目が不正解またはタイムアウトのため、もう一度音声全体を提示して再試行します。")
            used_hint = True

            silence_padding = np.zeros(int(FS * 0.2), dtype="float32")
            click_sound = generate_click_sound()

            # --- 録音2 (2回目 / 全音声再提示) ---
            _, timed_out_2, onset_2, aborted_2, audio_buffer_2 = run_single_recording_gui(
                audio_to_play=stimulus_audio,
                play_fs=stimulus_fs,
                window_title=f"復唱課題 - 試行 {index} (再提示による再試行)",
                prompt_text="【再】聞こえた文をそのまま繰り返してください。",
                listen_text="【再】聞いてください。",
                max_duration=max_response_duration
            )

            if aborted_2:
                print("[中断] ユーザーによって課題が中断されました。")
                task_aborted = True
                break

            if audio_buffer_2:
                try:
                    audio_array_2 = np.concatenate(audio_buffer_2, axis=0)
                except Exception as e:
                    print(f"[エラー] 配列結合失敗(2回目): {e}")
                    audio_array_2 = np.array([], dtype="float32")

                audio_array_2 = np.squeeze(audio_array_2)

                temp_audio_2 = os.path.join(OUTPUT_DIR, f"temp_trial{trial_num:02d}_2.wav")
                sf.write(temp_audio_2, audio_array_2, FS)
                recognized_text_2 = transcribe_audio_file(temp_audio_2, WHISPER_MODEL)
                analysis_2 = analyze_repetition(reference_sentence, recognized_text_2)
                
                final_audio_array = np.concatenate([
                    audio_array_1, 
                    silence_padding, 
                    click_sound, 
                    silence_padding, 
                    audio_array_2
                ], axis=0)
                
                if os.path.exists(temp_audio_2):
                    os.remove(temp_audio_2)

            if analysis_2["文全体の一致"] == "正解":
                trial_status = "ヒント後正解"
            else:
                trial_status = "ヒント後不正解"
        else:
            if not enable_hints:
                display_t1_judgment = "タイムアウト" if timed_out_1 else analysis_1["文全体の一致"]
                trial_status = "一発正解" if display_t1_judgment == "正解" else ("一発タイムアウト" if timed_out_1 else "一発不正解")
                trial_status += "（ヒントなしモード）"
            else:
                trial_status = "1回目正解"

        if os.path.exists(temp_audio_1):
            os.remove(temp_audio_1)

        final_audio_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_trial{trial_num:02d}_response.wav")
        sf.write(final_audio_file, final_audio_array, FS)

        final_analysis = analysis_2 if used_hint else analysis_1

        result_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_trial{trial_num:02d}_result.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"試行番号: {trial_num}\n")
            f.write(f"提示文: {reference_sentence}\n")
            f.write(f"使用モデル: {selected_model_name}\n")
            f.write(f"最大制限時間_s: {max_response_duration}\n")
            f.write(f"試行ステータス: {trial_status}\n")
            f.write(f"1回目 認識結果: {recognized_text_1} (判定: {analysis_1['文全体の一致']}, 反応時間: {f'{onset_1:.3f}秒' if onset_1 is not None else '無発話'})\n")
            if used_hint:
                f.write(f"2回目(再提示後) 認識結果: {recognized_text_2} (判定: {analysis_2['文全体の一致']}, 反応時間: {f'{onset_2:.3f}秒' if onset_2 is not None else '無発話'})\n")
            f.write(f"最終判定: {final_analysis['文全体の一致']}\n")

        all_results.append({
            "trial": trial_num,
            "stimulus_file": stimulus_filename,
            "audio_file": final_audio_file,
            "result_file": result_file,
            "analysis_1": analysis_1,
            "onset_1": onset_1,
            "analysis_2": analysis_2 if used_hint else {"Whisper認識結果": "", "文全体の一致": "", "提示語再現率": ""},
            "onset_2": onset_2,
            "final_analysis": final_analysis,
            "used_hint": used_hint,
            "trial_status": trial_status,
            "max_duration_sec": max_response_duration
        })

    if task_aborted:
        print("\n[中断] 課題が途中で終了されました（中断までに完了した試行のみ保存されます）。")

    # --------------------------------------------------------
    # 統合CSVの保存
    # --------------------------------------------------------
    if all_results:
        excel_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_results.csv")
        with open(excel_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
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
                "統合録音ファイル"
            ])
            for res in all_results:
                writer.writerow([
                    SUBJECT_ID,
                    res["trial"],
                    res["final_analysis"]["提示文"],
                    selected_model_name,
                    res["max_duration_sec"],
                    res["analysis_1"]["Whisper認識結果"],
                    res["analysis_1"]["文全体の一致"],
                    f"{res['onset_1']:.3f}" if res["onset_1"] is not None else "",
                    "あり" if res["used_hint"] else "なし",
                    res["analysis_2"]["Whisper認識結果"],
                    res["analysis_2"]["文全体の一致"],
                    f"{res['onset_2']:.3f}" if res["onset_2"] is not None else "",
                    res["final_analysis"]["文全体の一致"],
                    res["trial_status"],
                    os.path.basename(res["audio_file"])
                ])

        print(f"\n[完了] 出力先: {OUTPUT_DIR}")
        print(f"統合CSV: {excel_file}")
    else:
        print("\n[終了] 保存された試行結果はありません。")