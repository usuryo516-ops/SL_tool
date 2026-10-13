# ============================================================
# Naming.py
# ============================================================

import os
import sys
import json
import csv
import time
import datetime
import threading
import subprocess
import winsound
import ctypes

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import whisper

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


# ============================================================
# 基本設定
# ============================================================

WHISPER_MODEL = None

# Escape押下時に、課題全体を終了するための共有フラグ
TASK_ABORTED = threading.Event()

THRESHOLD = 0.08
FS = 44100
MAX_DURATION = 15  # デフォルト制限時間（設定画面で変更可能）[cite: 3]

# 録音開始直前の語頭を保護する先行バッファ長（秒）
PRE_ROLL_SEC = 0.0

DEFAULT_WHISPER_MODEL_NAME = "turbo"
WHISPER_MODEL_NAME = None

# ヒント設定のデフォルト（True: あり, False: なし）
DEFAULT_HAS_CUE = True
HAS_CUE_SETTING = True

CLICK_DURATION = 0.03
CLICK_FREQUENCY = 1000
CLICK_VOLUME = 0.70


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
IMG_FOLDER = os.path.join(SL_TOOL_DIR, "stimuli", "Naming")
MODEL_DIR = os.path.join(SL_TOOL_DIR, "whisper_model")

SYNONYMS_JSON_PATH = os.path.join(APP_DIR, "synonyms.json")
CUE_JSON_PATH = os.path.join(APP_DIR, "cue.json")

os.environ["PATH"] = (
    SL_TOOL_DIR
    + os.pathsep
    + os.environ.get("PATH", "")
)


# ============================================================
# 同義語辞書・キュー読み込み
# ============================================================

DEFAULT_COMMON_SYNONYMS = {
    "みかん": ["オレンジ", "柑子", "橙"],
    "オレンジ": ["みかん", "橙"],
    "いちご": ["ストロベリー"],
    "ストロベリー": ["いちご"],
    "ぶどう": ["グレープ"],
    "グレープ": ["ぶどう"],
    "もも": ["ピーチ"],
    "ピーチ": ["もも"],
    "すいか": ["ウォーターメロン"],
    "なし": ["ペアー", "梨"],
    "かき": ["パーシモン", "柿"]
}


def load_common_synonyms():
    synonyms = {
        key: list(values)
        for key, values in DEFAULT_COMMON_SYNONYMS.items()
    }
    if not os.path.exists(SYNONYMS_JSON_PATH):
        return synonyms
    try:
        with open(SYNONYMS_JSON_PATH, "r", encoding="utf-8-sig") as f:
            loaded_data = json.load(f)
        if isinstance(loaded_data, dict):
            for answer_word, synonym_list in loaded_data.items():
                if isinstance(answer_word, str) and isinstance(synonym_list, list):
                    valid_synonyms = [item.strip() for item in synonym_list if isinstance(item, str) and item.strip()]
                    synonyms[answer_word.strip()] = valid_synonyms
    except Exception as e:
        print(f"[警告] synonyms.json 読み込み失敗: {e}")
    return synonyms


def load_cues():
    if not os.path.exists(CUE_JSON_PATH):
        return {}
    try:
        with open(CUE_JSON_PATH, "r", encoding="utf-8-sig") as f:
            loaded_data = json.load(f)
        if isinstance(loaded_data, dict):
            cues = {}
            for answer_word, cue_text in loaded_data.items():
                if isinstance(answer_word, str) and isinstance(cue_text, str):
                    aw = answer_word.strip()
                    ct = cue_text.strip()
                    if aw and ct:
                        cues[aw] = ct
            return cues
    except Exception as e:
        print(f"[警告] cue.json 読み込み失敗: {e}")
    return {}


COMMON_SYNONYMS = load_common_synonyms()
CUES = load_cues()


# ============================================================
# 文字処理・判定
# ============================================================

def to_hiragana(text):
    katakana_map = str.maketrans({
        "ア": "あ", "イ": "い", "ウ": "う", "エ": "え", "オ": "お",
        "カ": "か", "キ": "き", "ク": "く", "ケ": "け", "コ": "こ",
        "サ": "さ", "シ": "し", "ス": "す", "セ": "せ", "ソ": "そ",
        "タ": "た", "チ": "ち", "ツ": "つ", "テ": "て", "ト": "と",
        "ナ": "な", "ニ": "に", "ヌ": "ぬ", "ネ": "ね", "ノ": "の",
        "ハ": "は", "ヒ": "ひ", "フ": "ふ", "ヘ": "へ", "ホ": "ほ",
        "マ": "ま", "ミ": "み", "ム": "む", "メ": "め", "モ": "も",
        "ヤ": "や", "ユ": "ゆ", "ヨ": "よ",
        "ラ": "ら", "リ": "り", "ル": "る", "レ": "れ", "ロ": "ろ",
        "ワ": "わ", "ヲ": "を", "ン": "ん",
        "ガ": "が", "ギ": "ぎ", "グ": "ぐ", "ゲ": "げ", "ゴ": "ご",
        "ザ": "ざ", "ジ": "じ", "ズ": "ず", "ゼ": "ぜ", "ゾ": "ぞ",
        "ダ": "だ", "ヂ": "ぢ", "ヅ": "づ", "デ": "で", "ド": "ど",
        "バ": "ば", "ビ": "び", "ブ": "ぶ", "ベ": "べ", "ボ": "ぼ",
        "パ": "ぱ", "ピ": "ぴ", "プ": "ぷ", "ペ": "ぺ", "ポ": "ぽ",
        "ァ": "ぁ", "ィ": "ぃ", "ゥ": "ぅ", "ェ": "ぇ", "ォ": "ぉ",
        "ャ": "ゃ", "ュ": "ゅ", "ョ": "ょ", "ッ": "っ", "ヴ": "ゔ"
    })
    return text.translate(katakana_map)


def get_initial_sound(word):
    word = word.strip()
    if not word:
        return ""
    return to_hiragana(word)[0]


def get_cue(answer_word):
    answer_word = answer_word.strip()
    if answer_word in CUES:
        return "word", CUES[answer_word]
    return "initial", get_initial_sound(answer_word)


def check_answer(recognized_text, answer_word):
    if not recognized_text:
        return False
    recognized_text = recognized_text.strip()
    answer_word = answer_word.strip()
    if answer_word in recognized_text:
        return True
    recognized_hira = to_hiragana(recognized_text)
    answer_hira = to_hiragana(answer_word)
    if answer_hira in recognized_hira:
        return True
    for synonym in COMMON_SYNONYMS.get(answer_word, []):
        if synonym in recognized_text:
            return True
        if to_hiragana(synonym) in recognized_hira:
            return True
    return False


# ============================================================
# TTS音声生成（音量増幅対応）
# ============================================================

def generate_tts_wav(text, output_path):
    output_path = os.path.abspath(output_path)
    tts_text = f"{text}。" if len(text) <= 1 else text
    success = False

    if PYTTSX3_AVAILABLE:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)
            engine.save_to_file(tts_text, output_path)
            engine.runAndWait()
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                success = True
        except Exception as e:
            print(f"[pyttsx3エラー] {e}")

    if not success:
        ps_script = f"""
        Add-Type -AssemblyName System.Speech
        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $synth.Volume = 100
        $synth.Rate = -1
        $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
        $synth.SetOutputToWaveFile('{output_path}', $format)
        $synth.Speak('{tts_text}')
        $synth.Dispose()
        """
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True, text=True, check=True, timeout=15
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                success = True
        except Exception as e:
            print(f"[TTSエラー] {e}")

    # 生成されたヒント音声の音量をソフトウェア的に増幅する
    if success:
        try:
            data, fs = sf.read(output_path)
            gain = 2.5  # 音量倍率（必要に応じて 2.0 〜 3.0 などに調整可能）
            data = data * gain
            data = np.clip(data, -1.0, 1.0)  # 音割れ（クリッピング）防止
            sf.write(output_path, data, fs)
        except Exception as e:
            print(f"[音量増幅エラー] {e}")

    return success


def create_click_audio():
    click_samples = max(1, int(FS * CLICK_DURATION))
    t = np.arange(click_samples) / FS
    click = CLICK_VOLUME * np.sin(2 * np.pi * CLICK_FREQUENCY * t)
    envelope = np.linspace(1.0, 0.0, click_samples)
    return (click * envelope).astype(np.float32).reshape(-1, 1)


# ============================================================
# ユーティリティ・GUI
# ============================================================

def safe_filename_part(text):
    invalid_chars = '\\/:*?"<>|'
    cleaned = "".join(char for char in text if char not in invalid_chars).strip()
    return cleaned if cleaned else "unknown"


def save_audio_file(audio, filename):
    if audio is None or len(audio) == 0:
        return False
    try:
        sf.write(filename, audio, FS)
        return True
    except Exception as e:
        print(f"[WAV保存エラー] {e}")
        return False




def recognize_from_file(filename):
    global WHISPER_MODEL
    result = WHISPER_MODEL.transcribe(
        filename, 
        language="ja", 
        task="transcribe", 
        temperature=[0.0, 0.2, 0.4],
        condition_on_previous_text=False,  # 前の文脈を引きずらない
        no_speech_threshold=0.6,          # 無音と判定するしきい値（高いほど無音を無視しやすくなる）
        compression_ratio_threshold=2.0   # 繰り返しパターンのハルシネーションを抑制
    )
    return result["text"].strip()


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


def abort_entire_task(event=None):
    TASK_ABORTED.set()
    return "break"


def select_whisper_model():
    root = tk.Tk()
    root.title("Whisperモデル選択")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    selected_model = tk.StringVar(value=DEFAULT_WHISPER_MODEL_NAME)
    result = [None]
    is_destroyed = [False]

    tk.Label(root, text="音声認識モデルを選択してください", font=("Meiryo", 36), bg="lightblue", fg="black").pack(pady=(60, 25))

    tk.Radiobutton(root, text="Small\n標準モード：比較的高速で実用的な認識精度", variable=selected_model, value="small", font=("Meiryo", 24), bg="lightblue", fg="black", selectcolor="white", justify="left", anchor="w", padx=30, pady=15).pack(fill=tk.X, padx=250, pady=5)
    tk.Radiobutton(root, text="Turbo\n高速モード：Enter後の待機時間を短くしたい場合", variable=selected_model, value="turbo", font=("Meiryo", 24), bg="lightblue", fg="black", selectcolor="white", justify="left", anchor="w", padx=30, pady=15).pack(fill=tk.X, padx=250, pady=5)
    tk.Radiobutton(root, text="Large\n高精度モード：待ち時間が長くても精度を優先", variable=selected_model, value="large", font=("Meiryo", 24), bg="lightblue", fg="black", selectcolor="white", justify="left", anchor="w", padx=30, pady=15).pack(fill=tk.X, padx=250, pady=5)

    tk.Label(root, text="↑↓: モデル選択    1: Small    2: Turbo    3: Large\nEnter または開始ボタンで次へ\nEscape で終了", font=("Meiryo", 18), bg="lightblue", fg="black", justify="center").pack(pady=25)

    def start_task(event=None):
        result[0] = selected_model.get()
        is_destroyed[0] = True
        try: root.destroy()
        except: pass
        return "break"

    def cancel_task(event=None):
        TASK_ABORTED.set()
        result[0] = None
        is_destroyed[0] = True
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

    while result[0] is None and not is_destroyed[0]:
        try:
            if not root.winfo_exists():
                break
            root.update()
            root.update_idletasks()
        except:
            break
        time.sleep(0.01)

    try: root.destroy()
    except: pass
    
    if TASK_ABORTED.is_set() or result[0] is None:
        sys.exit(0)

    return result[0]


def select_cue_setting():
    root = tk.Tk()
    root.title("ヒント設定選択")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    selected_cue = tk.BooleanVar(value=DEFAULT_HAS_CUE)
    result = [None]
    is_destroyed = [False]

    tk.Label(root, text="ヒントのありなしを選択してください", font=("Meiryo", 36), bg="lightblue", fg="black").pack(pady=(80, 40))

    tk.Radiobutton(root, text="ヒントあり\n不正解時にヒント音声を提示して2回目の試行を行います", variable=selected_cue, value=True, font=("Meiryo", 24), bg="lightblue", fg="black", selectcolor="white", justify="left", anchor="w", padx=30, pady=20).pack(fill=tk.X, padx=250, pady=10)
    tk.Radiobutton(root, text="ヒントなし\n不正解時もヒント音声を提示せず、1回のみ（またはヒントなしで進行）", variable=selected_cue, value=False, font=("Meiryo", 24), bg="lightblue", fg="black", selectcolor="white", justify="left", anchor="w", padx=30, pady=20).pack(fill=tk.X, padx=250, pady=10)

    tk.Label(root, text="↑↓: 選択    1: ヒントあり    2: ヒントなし\nEnter または開始ボタンで次へ\nEscape で終了", font=("Meiryo", 18), bg="lightblue", fg="black", justify="center").pack(pady=30)

    def start_task(event=None):
        result[0] = selected_cue.get()
        is_destroyed[0] = True
        try: root.destroy()
        except: pass
        return "break"

    def cancel_task(event=None):
        TASK_ABORTED.set()
        result[0] = None
        is_destroyed[0] = True
        try: root.destroy()
        except: pass
        return "break"

    cue_order = [True, False]
    def move_cue_selection(direction):
        curr = selected_cue.get()
        idx = cue_order.index(curr) if curr in cue_order else 0
        new_idx = max(0, min(idx + direction, len(cue_order) - 1))
        selected_cue.set(cue_order[new_idx])
        return "break"

    tk.Button(root, text="次へ", font=("Meiryo", 28), bg="white", fg="black", command=start_task, padx=70, pady=12).pack(pady=15)

    root.bind("<Return>", start_task)
    root.bind("<KP_Enter>", start_task)
    root.bind("<Escape>", cancel_task)
    root.bind("<Up>", lambda e: move_cue_selection(-1))
    root.bind("<Down>", lambda e: move_cue_selection(1))
    root.bind("1", lambda e: selected_cue.set(True))
    root.bind("2", lambda e: selected_cue.set(False))

    root.update_idletasks()
    root.update()
    force_window_focus(root)

    while result[0] is None and not is_destroyed[0]:
        try:
            if not root.winfo_exists():
                break
            root.update()
            root.update_idletasks()
        except:
            break
        time.sleep(0.01)

    try: root.destroy()
    except: pass
    
    if TASK_ABORTED.is_set() or result[0] is None:
        sys.exit(0)

    return result[0]


def select_duration_setting():
    root = tk.Tk()
    root.title("制限時間設定")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    duration_var = tk.StringVar(value=str(MAX_DURATION))
    result = [None]
    is_destroyed = [False]

    tk.Label(root, text="課題制限時間の設定", font=("Meiryo", 36), bg="lightblue", fg="black").pack(pady=(80, 40))
    tk.Label(root, text="1試行あたりの制限時間（秒）を半角数字で入力してください。\n（指定時間を超えると自動的に録音が終了します）", font=("Meiryo", 20), bg="lightblue", fg="darkblue", justify="center").pack(pady=(0, 30))

    input_frame = tk.Frame(root, bg="white", padx=50, pady=30)
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

    tk.Label(root, text="Enter またはボタンで課題を開始\nEscape で終了", font=("Meiryo", 18), bg="lightblue", fg="black", justify="center").pack(pady=30)

    def start_task(event=None):
        try:
            val = int(duration_var.get().strip())
            if val <= 0:
                raise ValueError
            result[0] = val
            is_destroyed[0] = True
            try: root.destroy()
            except: pass
        except ValueError:
            messagebox.showerror("入力エラー", "制限時間は正の整数で入力してください。")
        return "break"

    def cancel_task(event=None):
        TASK_ABORTED.set()
        result[0] = None
        is_destroyed[0] = True
        try: root.destroy()
        except: pass
        return "break"

    tk.Button(root, text="課題開始", font=("Meiryo", 28), bg="white", fg="black", command=start_task, padx=70, pady=12).pack(pady=15)

    root.bind("<Return>", start_task)
    root.bind("<KP_Enter>", start_task)
    root.bind("<Escape>", cancel_task)

    root.update_idletasks()
    root.update()
    force_window_focus(root)

    while result[0] is None and not is_destroyed[0]:
        try:
            if not root.winfo_exists():
                break
            root.update()
            root.update_idletasks()
        except:
            break
        time.sleep(0.01)

    try: root.destroy()
    except: pass
    
    if TASK_ABORTED.is_set() or result[0] is None:
        sys.exit(0)

    return result[0]


def show_instruction_and_wait(text, font_size=36):
    root = tk.Tk()
    root.title("インストラクション")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    tk.Label(
        root,
        text=text,
        font=("Meiryo", font_size),
        bg="lightblue",
        fg="black",
        justify="center"
    ).pack(expand=True)

    closed = [False]

    def close_window(event=None):
        if closed[0]:
            return "break"
        closed[0] = True
        try:
            root.destroy()
        except Exception:
            pass
        return "break"

    def abort_instruction(event=None):
        TASK_ABORTED.set()
        closed[0] = True
        try:
            root.destroy()
        except Exception:
            pass
        return "break"

    root.bind("<Return>", close_window)
    root.bind("<KP_Enter>", close_window)
    root.bind("<space>", close_window)
    root.bind("<Escape>", abort_instruction)
    root.bind_all("<Escape>", abort_instruction)

    force_window_focus(root)

    while not closed[0] and root.winfo_exists() and not TASK_ABORTED.is_set():
        try:
            root.update()
            root.update_idletasks()
        except Exception:
            break
        time.sleep(0.01)

    try:
        root.destroy()
    except Exception:
        pass

    time.sleep(0.1)
    return not TASK_ABORTED.is_set()


# ============================================================
# 録音クラス（PersistentAudioRecorder）
# ============================================================

class PersistentAudioRecorder:
    def __init__(self):
        self.detected_event = threading.Event()
        self.recording_active = threading.Event()
        self.stop_stream = threading.Event()
        self.stream_ready = threading.Event()

        self.audio_buffer = []
        self.pre_roll_buffer = []
        self.buffer_lock = threading.Lock()
        self.pre_roll_samples = max(1, int(FS * PRE_ROLL_SEC))

        self.reaction_time = None
        self.start_time = None
        self.audio_error = None
        self.audio_thread = None

    def _audio_monitor(self):
        def callback(indata, frames, time_info, status):
            audio_block = indata.copy()
            with self.buffer_lock:
                self.pre_roll_buffer.append(audio_block)
                total_samples = sum(len(block) for block in self.pre_roll_buffer)
                while self.pre_roll_buffer and total_samples > self.pre_roll_samples:
                    removed = self.pre_roll_buffer.pop(0)
                    total_samples -= len(removed)

                if self.recording_active.is_set():
                    self.audio_buffer.append(audio_block)

            if self.recording_active.is_set() and not self.detected_event.is_set() and self.start_time is not None:
                volume = float(np.abs(audio_block).max())
                if volume > THRESHOLD:
                    self.reaction_time = time.perf_counter() - self.start_time
                    self.detected_event.set()
                    print(f"[音声検出] RT={self.reaction_time:.3f} 秒")

        try:
            with sd.InputStream(samplerate=FS, channels=1, callback=callback, blocksize=256, dtype="float32"):
                self.stream_ready.set()
                while not self.stop_stream.is_set():
                    time.sleep(0.002)
        except Exception as e:
            self.audio_error = e
            self.stream_ready.set()

    def start(self):
        self.audio_thread = threading.Thread(target=self._audio_monitor, daemon=True)
        self.audio_thread.start()
        self.stream_ready.wait(timeout=3)
        return self.audio_error is None

    def begin_attempt(self, attempt_number):
        if self.audio_error is not None:
            return False
        with self.buffer_lock:
            self.audio_buffer = [block.copy() for block in self.pre_roll_buffer]
        self.detected_event.clear()
        self.reaction_time = None
        self.start_time = time.perf_counter()
        self.recording_active.set()
        print(f"[録音開始] attempt={attempt_number}")
        return True

    def end_attempt(self):
        self.recording_active.clear()
        with self.buffer_lock:
            if not self.audio_buffer:
                return None
            try:
                return np.concatenate(self.audio_buffer, axis=0)
            except Exception:
                return None

    def get_reaction_time(self):
        return self.reaction_time

    def close(self):
        self.recording_active.clear()
        self.stop_stream.set()
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=3)


def record_attempt_in_window(root, attempt_number, recorder, max_duration):
    finished = [False]
    if not recorder.begin_attempt(attempt_number):
        return None, None, False, True

    def finish_recording(reason):
        if finished[0]: return
        finished[0] = True
        recorder.recording_active.clear()

    def end_recording(event=None):
        if recorder.start_time is not None and time.perf_counter() - recorder.start_time < 0.2:
            return "break"
        finish_recording("key")
        return "break"

    def abort_recording(event=None):
        TASK_ABORTED.set()
        finish_recording("abort")
        return "break"

    root.bind("<Return>", end_recording)
    root.bind("<KP_Enter>", end_recording)
    root.bind("<space>", end_recording)
    root.bind("<Escape>", abort_recording)
    root.bind_all("<Return>", end_recording)
    root.bind_all("<KP_Enter>", end_recording)
    root.bind_all("<space>", end_recording)
    root.bind_all("<Escape>", abort_recording)

    force_window_focus(root)

    while not finished[0] and root.winfo_exists() and not TASK_ABORTED.is_set():
        try:
            root.update()
            root.update_idletasks()
        except:
            break
        if recorder.start_time is not None and time.perf_counter() - recorder.start_time >= max_duration:
            finish_recording("timeout")
            break
        time.sleep(0.01)

    audio_data = recorder.end_attempt()
    reaction_time = recorder.get_reaction_time()
    return audio_data, reaction_time, finished[0] and (recorder.start_time is not None and time.perf_counter() - recorder.start_time >= max_duration), False


# ============================================================
# 1試行の実行
# ============================================================

def run_trial_with_automatic_cue(image_path, trial_num, answer_word, output_dir, file_prefix, has_cue, max_duration):
    result = {
        "has_cue": "あり" if has_cue else "なし",
        "max_duration_sec": max_duration,
        "attempt1_rt_sec": "", "attempt1_result": "", "attempt1_recognized": "", "attempt1_timeout": "",
        "cue_type": "", "cue_text": "",
        "attempt2_rt_sec": "", "attempt2_result": "", "attempt2_recognized": "", "attempt2_timeout": "",
        "final_result": "", "final_recognized": "", "audio_file": ""
    }

    root = tk.Tk()
    root.title("刺激画像")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="white")

    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.copy()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        image.thumbnail((screen_width, screen_height), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(image)
        image_label = tk.Label(root, image=tk_image, bg="white")
        image_label.image = tk_image
        image_label.pack(expand=True, fill=tk.BOTH)
    except Exception as e:
        print(f"[画像エラー] {e}")
        try: root.destroy()
        except: pass
        result["attempt1_result"] = "画像エラー"
        result["final_result"] = "画像エラー"
        return result, False

    root.update_idletasks()
    root.update()
    force_window_focus(root)

    recorder = PersistentAudioRecorder()
    if not recorder.start():
        result["attempt1_result"] = "音声入力エラー"
        result["final_result"] = "音声入力エラー"
        try: root.destroy()
        except: pass
        return result, False

    # 1回目録音
    audio1, rt1, timeout1, error1 = record_attempt_in_window(root, 1, recorder, max_duration)

    if TASK_ABORTED.is_set():
        recorder.close()
        try: root.destroy()
        except: pass
        return result, False

    result["attempt1_rt_sec"] = f"{rt1:.3f}" if rt1 is not None else ""
    result["attempt1_timeout"] = timeout1

    recognized1 = ""
    result1 = ""
    safe_answer = safe_filename_part(answer_word)
    temp_wav1 = os.path.join(output_dir, f"_temp_{trial_num:02d}_attempt1.wav")

    if error1: result1 = "音声入力エラー"
    elif timeout1: result1 = "タイムアウト"
    elif audio1 is None: result1 = "反応なし"
    else:
        if save_audio_file(audio1, temp_wav1):
            try:
                print("1回目の音声認識中...")
                recognized1 = recognize_from_file(temp_wav1)
                result1 = "正解" if check_answer(recognized1, answer_word) else "不正解"
            except Exception:
                result1 = "音声認識エラー"

    result["attempt1_result"] = result1
    result["attempt1_recognized"] = recognized1
    print(f"1回目結果: {result1} ({recognized1})")

    if result1 == "正解" or not has_cue:
        final_wav = os.path.join(
            output_dir,
            f"{file_prefix}_{trial_num:02d}_{safe_answer}.wav"
        )
        if audio1 is not None:
            save_audio_file(audio1, final_wav)
            result["audio_file"] = final_wav

        result["final_result"] = result1
        result["final_recognized"] = recognized1

        if os.path.exists(temp_wav1):
            os.remove(temp_wav1)

        recorder.close()
        try:
            root.destroy()
        except Exception:
            pass

        return result, (result1 == "正解")

    # --- ヒントありの場合の2回目処理 ---
    cue_type, cue_text = get_cue(answer_word)
    result["cue_type"] = cue_type
    result["cue_text"] = cue_text
    print(f"[cue提示] type={cue_type}, text={cue_text}")

    cue_wav_path = os.path.join(output_dir, f"_temp_cue_{trial_num:02d}.wav")
    success_tts = generate_tts_wav(cue_text, cue_wav_path)

    # 2回目録音：ヒント提示同時スタート
    if not recorder.begin_attempt(2):
        result["attempt2_result"] = "音声入力エラー"
        result["final_result"] = "音声入力エラー"
        recorder.close()
        try: root.destroy()
        except: pass
        return result, False

    if success_tts and os.path.exists(cue_wav_path):
        winsound.PlaySound(cue_wav_path, winsound.SND_ASYNC | winsound.SND_FILENAME)
    else:
        print("[警告] ヒント音声ファイルがありません。")

    finished = [False]

    def finish_recording_2(reason):
        if finished[0]: return
        finished[0] = True
        recorder.recording_active.clear()

    def end_recording_2(event=None):
        if recorder.start_time is not None and time.perf_counter() - recorder.start_time < 0.2:
            return "break"
        finish_recording_2("key")
        return "break"

    def abort_recording_2(event=None):
        TASK_ABORTED.set()
        finish_recording_2("abort")
        return "break"

    root.bind("<Return>", end_recording_2)
    root.bind("<KP_Enter>", end_recording_2)
    root.bind("<space>", end_recording_2)
    root.bind("<Escape>", abort_recording_2)
    root.bind_all("<Return>", end_recording_2)
    root.bind_all("<KP_Enter>", end_recording_2)
    root.bind_all("<space>", end_recording_2)
    root.bind_all("<Escape>", abort_recording_2)

    force_window_focus(root)

    while not finished[0] and root.winfo_exists() and not TASK_ABORTED.is_set():
        try:
            root.update()
            root.update_idletasks()
        except:
            break
        if recorder.start_time is not None and time.perf_counter() - recorder.start_time >= max_duration:
            finish_recording_2("timeout")
            break
        time.sleep(0.01)

    audio2_mic = recorder.end_attempt()
    rt2_raw = recorder.get_reaction_time()
    timeout2 = finished[0] and (recorder.start_time is not None and time.perf_counter() - recorder.start_time >= max_duration)

    if TASK_ABORTED.is_set():
        recorder.close()
        try: root.destroy()
        except: pass
        return result, False

    if rt2_raw is not None:
        result["attempt2_rt_sec"] = f"{rt2_raw:.3f}"
    else:
        result["attempt2_rt_sec"] = ""

    result["attempt2_timeout"] = timeout2

    recognized2 = ""
    result2 = ""
    temp_wav2 = os.path.join(output_dir, f"_temp_{trial_num:02d}_attempt2.wav")

    if finished[0] and recorder.audio_error is not None:
        result2 = "音声入力エラー"
    elif timeout2:
        result2 = "タイムアウト"
    elif audio2_mic is None or len(audio2_mic) == 0:
        result2 = "反応なし"
    else:
        if save_audio_file(audio2_mic, temp_wav2):
            try:
                print("2回目の音声認識中...")
                recognized2 = recognize_from_file(temp_wav2)
                result2 = "正解" if check_answer(recognized2, answer_word) else "不正解"
            except Exception:
                result2 = "音声認識エラー"

    result["attempt2_result"] = result2
    result["attempt2_recognized"] = recognized2
    result["final_recognized"] = recognized2

    print(f"2回目結果: {result2} ({recognized2})")

    final_wav = os.path.join(output_dir, f"{file_prefix}_{trial_num:02d}_{safe_answer}.wav")
    if audio1 is not None and audio2_mic is not None:
        try:
            click_audio = create_click_audio()
            merged_audio = np.concatenate([audio1.flatten(), click_audio.flatten(), audio2_mic.flatten()], axis=0)
            if save_audio_file(merged_audio, final_wav):
                result["audio_file"] = final_wav
                print(f"[結合WAV保存完了] {final_wav}")
        except Exception as e:
            print(f"[結合保存エラー] {e}")

    for t_file in [temp_wav1, temp_wav2, cue_wav_path]:
        if t_file and os.path.exists(t_file):
            try: os.remove(t_file)
            except: pass

    if result2 == "正解":
        result["final_result"] = "cue後正解"
        is_correct = True
    elif result2 == "不正解":
        result["final_result"] = "cue後不正解"
        is_correct = False
    elif result2 == "タイムアウト":
        result["final_result"] = "cue後タイムアウト"
        is_correct = False
    else:
        result["final_result"] = f"cue後{result2}"
        is_correct = False

    recorder.close()

    try:
        root.destroy()
    except Exception:
        pass

    return result, is_correct


# ============================================================
# 画像ファイル取得・メイン
# ============================================================

def get_image_files(folder_path):
    files = []
    if not os.path.exists(folder_path): return files
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")) or "_" not in filename: continue
        num_part, rest = filename.split("_", 1)
        if not num_part.isdigit(): continue
        ans = None
        for ext in (".jpg", ".jpeg", ".png"):
            if rest.lower().endswith(ext):
                ans = rest[:-len(ext)]
                break
        if ans and ans.strip():
            files.append((int(num_part), filename, ans.strip()))
    files.sort(key=lambda r: r[0])
    return files


def create_empty_result_row(trial_num, filename, answer_word):
    return {
        "trial_num": trial_num, "image_file": filename, "answer_word": answer_word,
        "has_cue": "", "max_duration_sec": "",
        "attempt1_rt_sec": "", "attempt1_result": "", "attempt1_recognized": "", "attempt1_timeout": "",
        "cue_type": "", "cue_text": "", "attempt2_rt_sec": "", "attempt2_result": "", "attempt2_recognized": "", "attempt2_timeout": "",
        "final_result": "", "final_recognized": "", "audio_file": ""
    }


if __name__ == "__main__":
    WHISPER_MODEL_NAME = select_whisper_model()
    if WHISPER_MODEL_NAME is None or TASK_ABORTED.is_set():
        sys.exit(0)

    HAS_CUE_SETTING = select_cue_setting()
    if HAS_CUE_SETTING is None or TASK_ABORTED.is_set():
        sys.exit(0)

    MAX_DURATION = select_duration_setting()
    if MAX_DURATION is None or TASK_ABORTED.is_set():
        sys.exit(0)

    if len(sys.argv) < 2:
        print("使用方法: python Naming.py [被験者ID] [被験者名]")
        sys.exit(1)

    subject_id = sys.argv[1]
    subject_name = sys.argv[2] if len(sys.argv) >= 3 else ""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    subject_folder = f"{subject_id}_{subject_name}" if subject_name else subject_id
    subject_dir = os.path.join(RESULTS_DIR, subject_folder)
    os.makedirs(subject_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    hint_str = "hintON" if HAS_CUE_SETTING else "hintOFF"
    output_folder_name = f"Naming_{hint_str}_{timestamp}_{subject_folder}"
    
    output_dir = os.path.join(subject_dir, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    file_prefix = f"Naming_{timestamp}_{subject_folder}"

    image_files = get_image_files(IMG_FOLDER)
    if not image_files:
        print("[エラー] 刺激画像が見つかりません。")
        sys.exit(1)

    # インストラクション表示
    instruction_text = (
        "【 呼称課題 】\n\n"
        "画面に画像が表示されたら、名前を声を出して答えてください。\n"
        "回答が終わったら Enter キー または スペースキー を押してください。\n\n"
        "準備ができたら Enter キー を押して開始します。"
    )
    if not show_instruction_and_wait(instruction_text):
        sys.exit(0)

    print(f"Whisper {WHISPER_MODEL_NAME} ロード中...")
    WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME, download_root=MODEL_DIR)

    results = []
    correct_count = 0

    for index, (trial_num, filename, answer_word) in enumerate(image_files):
        if TASK_ABORTED.is_set():
            print("\n[中断] エスケープキーにより課題が中止されました。")
            break

        image_path = os.path.join(IMG_FOLDER, filename)
        print(f"\n--- 試行 {index + 1}/{len(image_files)}: {answer_word} ({filename}) ---")

        trial_result, is_correct = run_trial_with_automatic_cue(
            image_path=image_path,
            trial_num=trial_num,
            answer_word=answer_word,
            output_dir=output_dir,
            file_prefix=file_prefix,
            has_cue=HAS_CUE_SETTING,
            max_duration=MAX_DURATION
        )

        row = create_empty_result_row(trial_num, filename, answer_word)
        row.update(trial_result)
        results.append(row)

        if is_correct:
            correct_count += 1

        time.sleep(0.5)

    # 結果CSV出力
    csv_filename = os.path.join(output_dir, f"{file_prefix}_results.csv")
    fieldnames = [
        "trial_num", "image_file", "answer_word", "has_cue", "max_duration_sec",
        "attempt1_rt_sec", "attempt1_result", "attempt1_recognized", "attempt1_timeout",
        "cue_type", "cue_text",
        "attempt2_rt_sec", "attempt2_result", "attempt2_recognized", "attempt2_timeout",
        "final_result", "final_recognized", "audio_file"
    ]

    try:
        with open(csv_filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[結果保存完了] {csv_filename}")
    except Exception as e:
        print(f"\n[CSV保存エラー] {e}")

    # 終了画面表示
    completion_text = (
        f"全試行が終了しました。\n\n"
        f"正解数: {correct_count} / {len(image_files)}\n\n"
        f"Enter キーを押して終了します。"
    )
    show_instruction_and_wait(completion_text, font_size=32)