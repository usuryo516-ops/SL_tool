import datetime
import os
import re
import sys
import time
import tkinter as tk

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

from janome.tokenizer import Tokenizer
from animal_dict import animal_dict


# ============================================================
# 基本設定
# ============================================================

FS = 44100
TASK_DURATION = 60
DEFAULT_WHISPER_MODEL_NAME = "turbo"

TASK_TITLE = "語想起課題"
CATEGORY_NAME = "動物"

INSTRUCTION_TEXT = (
    "【語想起課題】\n\n"
    "「動物」の名前を\n"
    "1 分間でできるだけ多く言ってください\n\n"
    "準備ができたら Enter キーを押してください\n"
    "（Escキーで終了）"
)

TASK_TEXT = (
    "「動物」の名前を\n"
    "1 分間でできるだけ多く言ってください"
)


# ============================================================
# SL_tool 共通パス設定
# ============================================================

def get_app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = get_app_dir()
SL_TOOL_DIR = os.path.dirname(APP_DIR)
RESULTS_DIR = os.path.join(SL_TOOL_DIR, "results")
MODEL_DIR = os.path.join(SL_TOOL_DIR, "whisper_model")

os.environ["PATH"] = SL_TOOL_DIR + os.pathsep + os.environ.get("PATH", "")


# ============================================================
# Whisper モデル選択 GUI (フルスクリーン・上下選択対応)
# ============================================================

def select_model():
    root = tk.Tk()

    root.title("Whisper モデル選択")
    root.configure(bg="lightblue")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    selected_model = tk.StringVar(
        value=DEFAULT_WHISPER_MODEL_NAME
    )

    selection_result = [None]

    container = tk.Frame(root, bg="lightblue")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【Whisper 音声認識モデルの選択】",
        font=("Meiryo", 32, "bold"),
        bg="lightblue",
        fg="black"
    ).pack(
        pady=(0, 15)
    )

    tk.Label(
        container,
        text="文字起こしに使用するモデルを選択してください。\n（上下キーで選択変更、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightblue",
        fg="darkblue",
        justify="center"
    ).pack(
        pady=(0, 30)
    )

    model_frame = tk.Frame(
        container,
        bg="white",
        padx=40,
        pady=30
    )
    model_frame.pack(
        pady=10
    )

    model_options = [
        (
            "small",
            "Small",
            "速度と精度のバランスを重視"
        ),
        (
            "turbo",
            "Turbo",
            "高速処理向け"
        ),
        (
            "large",
            "Large",
            "高精度（初期選択）、処理時間・必要メモリは大きい"
        )
    ]

    for model_name, label, description in model_options:
        option_frame = tk.Frame(
            model_frame,
            bg="white"
        )
        option_frame.pack(
            fill="x",
            pady=10
        )

        rb = tk.Radiobutton(
            option_frame,
            text=label,
            variable=selected_model,
            value=model_name,
            font=("Meiryo", 18, "bold"),
            bg="white",
            activebackground="white",
            width=12,
            anchor="w"
        )
        rb.pack(
            side="left",
            padx=(0, 20)
        )

        tk.Label(
            option_frame,
            text=description,
            font=("Meiryo", 14),
            bg="white",
            fg="gray"
        ).pack(
            side="left"
        )

    button_frame = tk.Frame(
        container,
        bg="lightblue"
    )
    button_frame.pack(
        pady=40
    )

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
            current_idx = 2

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
    ).pack(
        side="left",
        padx=15
    )

    tk.Button(
        button_frame,
        text="キャンセル / 終了 (Esc)",
        command=cancel_selection,
        font=("Meiryo", 16),
        width=20,
        height=2,
        bg="white"
    ).pack(
        side="left",
        padx=15
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        cancel_selection
    )

    root.bind_all(
        "<Return>",
        start_selected_model
    )

    root.bind_all(
        "<Escape>",
        cancel_selection
    )

    root.lift()
    root.focus_force()
    root.focus_set()

    root.after(10, root.lift)
    root.after(50, root.focus_force)
    root.after(100, root.focus_set)

    root.mainloop()

    return selection_result[0]


# ============================================================
# オーディオデバイス情報
# ============================================================

def show_audio_devices():
    print("\n" + "=" * 60)
    print("利用可能なオーディオ入力デバイス")
    print("=" * 60)

    devices = sd.query_devices()

    for index, device in enumerate(devices):
        input_channels = device["max_input_channels"]

        if input_channels > 0:
            print(
                f"{index}: {device['name']} "
                f"（入力チャンネル: {input_channels}）"
            )

    default_input = sd.query_devices(kind="input")

    print("")
    print(f"デフォルト入力デバイス: {default_input['name']}")
    print(f"サンプリングレート: {default_input['default_samplerate']} Hz")
    print(f"入力チャンネル数: {default_input['max_input_channels']}")
    print("=" * 60 + "\n")

    return default_input


# ============================================================
# 開始インストラクション
# ============================================================

def show_start_instruction():
    root = tk.Tk()

    root.title("語想起課題 - 開始インストラクション")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    root.focus_force()
    root.lift()
    root.focus_set()

    label = tk.Label(
        root,
        text=INSTRUCTION_TEXT,
        font=("Meiryo", 36, "bold"),
        bg="lightblue",
        fg="black",
        justify="center"
    )
    label.pack(expand=True)

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

    root.bind("<Return>", start_task)
    root.bind("<space>", start_task)
    root.bind("<Escape>", cancel_task)
    root.bind_all("<Escape>", cancel_task)

    root.after(10, lambda: root.focus_force())
    root.after(50, lambda: root.lift())
    root.after(100, lambda: root.focus_set())

    root.mainloop()

    time.sleep(0.1)

    return started[0]


# ============================================================
# 1分間録音課題
# ============================================================

def show_task_and_record_gui(output_filename, fs=FS, duration=TASK_DURATION):
    root = tk.Tk()

    root.title("語想起課題 - 動物")

    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="white")

    root.focus_force()
    root.lift()
    root.focus_set()

    instruction_label = tk.Label(
        root,
        text=TASK_TEXT,
        font=("Meiryo", 48, "bold"),
        fg="black",
        bg="white",
        justify="center"
    )
    instruction_label.pack(expand=True, pady=(100, 50))

    timer_label = tk.Label(
        root,
        text=f"残り時間: {duration} 秒",
        font=("Meiryo", 72, "bold"),
        fg="red",
        bg="lightyellow",
        pady=30
    )
    timer_label.pack(fill=tk.X, side=tk.BOTTOM)

    status_label = tk.Label(
        root,
        text="【準備中】Enter キーで開始 / Esc キーで中止",
        font=("Meiryo", 28, "bold"),
        fg="blue",
        bg="lightgreen",
        pady=20
    )
    status_label.pack(fill=tk.X, side=tk.BOTTOM)

    recorded_audio = [None]
    timer_value = [duration]
    recording_started = [False]
    task_finished = [False]
    canceled = [False]

    def update_timer():
        if not recording_started[0]:
            return

        if timer_value[0] > 0:
            timer_value[0] -= 1

            timer_label.config(
                text=f"残り時間: {timer_value[0]} 秒"
            )

            if timer_value[0] <= 10:
                timer_label.config(
                    fg="white",
                    bg="orange"
                )

            root.after(1000, update_timer)

        else:
            task_finished[0] = True

            timer_label.config(
                text="終了！",
                fg="green",
                bg="lightgreen"
            )

            status_label.config(
                text="【終了】しばらくお待ちください",
                fg="gray",
                bg="lightgray"
            )

            root.after(1500, root.quit)

    def start_recording_and_timer(event=None):
        if recording_started[0]:
            return

        recording_started[0] = True

        print(f"\n[録音開始] {duration} 秒間の録音を開始します。")

        try:
            input_device = sd.query_devices(kind="input")
            print(f"入力デバイス: {input_device['name']}")
            print(f"サンプリングレート: {fs} Hz")
        except Exception as e:
            print(f"[警告] 入力デバイス情報を取得できませんでした: {e}")

        status_label.config(
            text="【録音中】動物の名前を言ってください（Escキーで中止）",
            fg="white",
            bg="red"
        )

        try:
            recorded_audio[0] = sd.rec(
                int(duration * fs),
                samplerate=fs,
                channels=1,
                dtype="float32"
            )
        except Exception as e:
            print(f"[録音開始エラー] {e}")

            status_label.config(
                text="【エラー】録音を開始できませんでした",
                fg="white",
                bg="red"
            )

            root.after(2000, root.quit)
            return

        root.after(1000, update_timer)

    def cancel_task(event=None):
        canceled[0] = True
        try:
            sd.stop()
        except Exception:
            pass
        try:
            root.quit()
        except Exception:
            pass

    def keep_focus():
        try:
            if root.winfo_exists() and not task_finished[0]:
                root.focus_force()
                root.after(250, keep_focus)
        except Exception:
            pass

    root.bind("<Return>", start_recording_and_timer)
    root.bind("<space>", start_recording_and_timer)
    root.bind("<Escape>", cancel_task)
    root.bind_all("<Escape>", cancel_task)

    root.after(10, lambda: root.focus_force())
    root.after(50, lambda: root.lift())
    root.after(100, lambda: root.focus_set())
    root.after(250, keep_focus)

    root.mainloop()

    try:
        root.destroy()
    except Exception:
        pass

    if canceled[0]:
        print("[キャンセル] 課題が中止されました。")
        return None

    if recorded_audio[0] is None:
        print("[エラー] 録音が開始されませんでした。")
        return None

    try:
        sd.wait()
    except Exception as e:
        print(f"[録音待機エラー] {e}")
        return None

    print("[録音完了]")

    output_filename = os.path.normpath(output_filename)
    output_dir = os.path.dirname(output_filename)

    if not os.path.exists(output_dir):
        print(f"[エラー] 出力フォルダーが存在しません: {output_dir}")
        return None

    try:
        sf.write(
            output_filename,
            recorded_audio[0],
            fs
        )
    except Exception as e:
        print(f"[音声保存エラー] {e}")
        return None

    time.sleep(0.3)

    if not os.path.exists(output_filename):
        print(f"[エラー] WAV ファイルが保存されませんでした: {output_filename}")
        return None

    file_size = os.path.getsize(output_filename)
    audio_duration = len(recorded_audio[0]) / fs
    rms = np.sqrt(np.mean(recorded_audio[0] ** 2))

    print(f"[音声保存完了] {output_filename}")
    print(f"ファイルサイズ: {file_size} bytes")
    print(f"録音時間: {audio_duration:.2f} 秒")
    print(f"録音レベル RMS: {rms:.5f}")

    if rms < 0.001:
        print("")
        print("[警告] 録音レベルが非常に低い可能性があります。")
        print("マイク入力デバイス、音量、Windows のマイク権限を確認してください。")

    return output_filename


# ============================================================
# Whisper による文字起こし
# ============================================================

def transcribe_audio(audio_path, model):
    audio_path = os.path.normpath(audio_path)

    if not os.path.exists(audio_path):
        print(f"[エラー] 音声ファイルが見つかりません: {audio_path}")
        return {
            "text": "",
            "utterances": []
        }

    print("")
    print("==================================================")
    print("Whisper による文字起こしを開始します")
    print("==================================================")
    print(f"音声ファイル: {audio_path}")

    try:
        # OpenAI公式Whisperがサポートする標準パラメータのみを使用
        result = model.transcribe(
            audio_path,
            language="ja",
            task="transcribe",
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.0,
        )
    except Exception as e:
        print(f"[エラー] Whisper の文字起こしに失敗しました: {e}")

        return {
            "text": "",
            "utterances": []
        }

    segments = result.get("segments", [])

    utterances = [
        segment["text"].strip()
        for segment in segments
        if segment.get("text", "").strip()
    ]

    full_text = " ".join(utterances).strip()

    default_phrases = [
        "ご視聴ありがとうございました",
        "ありがとうございました",
        "ご視聴",
        "Thank you for watching",
        "Thanks for watching",
    ]

    for phrase in default_phrases:
        full_text = full_text.replace(phrase, "")

    # --- 無音ループ（同じ単語の異常な連続）を自動カットする処理 ---
    full_text = re.sub(r'([一-龥ぁ-んァ-ン]+)(?:\1){4,}', '', full_text)

    full_text = full_text.strip()

    print("")
    print("文字起こし結果:")
    print(full_text if full_text else "（認識結果なし）")

    return {
        "text": full_text,
        "utterances": utterances
    }


# ============================================================
# テキストの正規化
# ============================================================

def normalize_text_for_matching(text):
    text = text.replace(" ", "")
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    text = text.replace("\t", "")

    return text


# ============================================================
# 動物名抽出・分析
# ============================================================

def build_surface_to_kana_dict():
    surface_to_kana = {}

    for kana_representation, surfaces in animal_dict.items():
        for surface in surfaces:
            normalized_surface = normalize_text_for_matching(surface)

            if normalized_surface:
                surface_to_kana[normalized_surface] = kana_representation

        normalized_kana = normalize_text_for_matching(kana_representation)

        if normalized_kana:
            surface_to_kana[normalized_kana] = kana_representation

    return surface_to_kana


def extract_animal_words(full_text):
    normalized_text = normalize_text_for_matching(full_text)

    if not normalized_text:
        return [], [], normalized_text

    surface_to_kana = build_surface_to_kana_dict()

    sorted_surfaces = sorted(
        surface_to_kana.keys(),
        key=len,
        reverse=True
    )

    matched_spans = []

    for surface in sorted_surfaces:
        search_start = 0

        while True:
            found_index = normalized_text.find(surface, search_start)

            if found_index == -1:
                break

            found_end = found_index + len(surface)

            overlap = False

            for existing_start, existing_end, _ in matched_spans:
                if found_index < existing_end and found_end > existing_start:
                    overlap = True
                    break

            if not overlap:
                matched_spans.append(
                    (
                        found_index,
                        found_end,
                        surface_to_kana[surface]
                    )
                )

            search_start = found_end

    matched_spans.sort(key=lambda item: item[0])

    animal_words = [
        kana_word
        for _, _, kana_word in matched_spans
    ]

    return animal_words, matched_spans, normalized_text


def remove_adjacent_duplicates(words):
    cleaned_words = []
    previous_word = None

    for word in words:
        if word != previous_word:
            cleaned_words.append(word)

        previous_word = word

    return cleaned_words


def get_unique_words_in_order(words):
    unique_words = []
    seen = set()

    for word in words:
        if word not in seen:
            unique_words.append(word)
            seen.add(word)

    return unique_words


def extract_nouns_for_review(text, matched_spans):
    tokenizer = Tokenizer()

    nouns = []

    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(",")[0]

        if pos == "名詞":
            word = token.surface.strip()

            if word:
                nouns.append(word)

    normalized_text = normalize_text_for_matching(text)

    animal_ranges = [
        (start, end)
        for start, end, _ in matched_spans
    ]

    review_words = []
    search_start = 0

    for noun in nouns:
        normalized_noun = normalize_text_for_matching(noun)

        if not normalized_noun:
            continue

        noun_index = normalized_text.find(
            normalized_noun,
            search_start
        )

        if noun_index >= 0:
            noun_end = noun_index + len(normalized_noun)
            search_start = noun_end
        else:
            noun_index = -1
            noun_end = -1

        inside_animal = False

        if noun_index >= 0:
            for animal_start, animal_end in animal_ranges:
                if noun_index >= animal_start and noun_end <= animal_end:
                    inside_animal = True
                    break

        if not inside_animal:
            review_words.append(noun)

    return nouns, review_words


def analyze_word_fluency(transcription_result):
    full_text = transcription_result.get("text", "").strip()
    utterances = transcription_result.get("utterances", [])

    if not full_text:
        return {
            "全文": "（無音または認識結果なし）",
            "Whisperセグメント数": 0,
            "抽出名詞数": 0,
            "動物名数（重複込み）": 0,
            "動物名数（隣接重複除去後）": 0,
            "動物名数（重複除く）": 0,
            "動物名一覧（発話順）": "（なし）",
            "動物名一覧（重複除く）": "（なし）",
            "要確認名詞": "（なし）",
        }

    animal_words_raw, matched_spans, _ = extract_animal_words(full_text)

    animal_words_cleaned = remove_adjacent_duplicates(animal_words_raw)

    unique_animal_words = get_unique_words_in_order(
        animal_words_cleaned
    )

    nouns, review_words = extract_nouns_for_review(
        full_text,
        matched_spans
    )

    print("")
    print("==================================================")
    print("語想起課題（動物）：分析結果")
    print("==================================================")

    print("")
    print(f"Whisper セグメント数: {len(utterances)}")
    print(f"抽出名詞数: {len(nouns)}")

    print("")
    print(f"動物名（重複込み）: {len(animal_words_raw)} 語")
    print(
        "・".join(animal_words_raw)
        if animal_words_raw else "（なし）"
    )

    print("")
    print(f"動物名（隣接重複除去後）: {len(animal_words_cleaned)} 語")
    print(
        "・".join(animal_words_cleaned)
        if animal_words_cleaned else "（なし）"
    )

    print("")
    print(f"動物名（重複除く）: {len(unique_animal_words)} 語")
    print(
        "・".join(unique_animal_words)
        if unique_animal_words else "（なし）"
    )

    print("")
    print(f"要確認名詞: {len(review_words)} 語")
    print(
        "・".join(review_words)
        if review_words else "（なし）"
    )

    return {
        "全文": full_text,
        "Whisperセグメント数": len(utterances),
        "抽出名詞数": len(nouns),
        "動物名数（重複込み）": len(animal_words_raw),
        "動物名数（隣接重複除去後）": len(animal_words_cleaned),
        "動物名数（重複除く）": len(unique_animal_words),
        "動物名一覧（発話順）": (
            "、".join(animal_words_cleaned)
            if animal_words_cleaned else "（なし）"
        ),
        "動物名一覧（重複除く）": (
            "、".join(unique_animal_words)
            if unique_animal_words else "（なし）"
        ),
        "要確認名詞": (
            "、".join(review_words)
            if review_words else "（なし）"
        ),
    }


# ============================================================
# TXT 保存
# ============================================================

def save_analysis_to_txt(analysis_results, output_filename):
    output_filename = os.path.normpath(output_filename)

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("        語想起課題（動物カテゴリー）分析結果\n")
            f.write("==================================================\n\n")

            for key, value in analysis_results.items():
                if key == "全文":
                    f.write(f"■ {key}\n")
                    f.write(f"{value}\n\n")
                else:
                    f.write(f"■ {key}: {value}\n")

        print(f"[保存完了] 分析結果: {output_filename}")

    except Exception as e:
        print(f"[エラー] 結果テキストの保存に失敗しました: {e}")


# ============================================================
# メイン処理
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("")
        print("使用方法:")
        print("python Word_Fluency.py [被験者ID] [被験者名]")
        print("")
        print("例:")
        print("python Word_Fluency.py 001 山田太郎")
        print("")
        print("被験者名を省略する場合:")
        print("python Word_Fluency.py 001")
        print("")
        sys.exit(1)

    SUBJECT_ID = sys.argv[1]
    SUBJECT_NAME = sys.argv[2] if len(sys.argv) >= 3 else ""

    print("")
    print("==================================================")
    print("Word Fluency 課題を開始します")
    print("==================================================")
    print(f"Word_Fluency フォルダー: {APP_DIR}")
    print(f"SL_tool フォルダー: {SL_TOOL_DIR}")
    print(f"結果フォルダー: {RESULTS_DIR}")
    print(f"Whisper モデルフォルダー: {MODEL_DIR}")
    print("==================================================")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Whisper モデル選択
    # --------------------------------------------------------

    WHISPER_MODEL_NAME = select_model()

    if WHISPER_MODEL_NAME is None:
        print("モデル選択がキャンセル（またはEscキーで終了）されました。")
        sys.exit(0)

    if SUBJECT_NAME:
        subject_folder = f"{SUBJECT_ID}_{SUBJECT_NAME}"
    else:
        subject_folder = SUBJECT_ID

    SUBJECT_DIR = os.path.join(
        RESULTS_DIR,
        subject_folder
    )

    os.makedirs(SUBJECT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if SUBJECT_NAME:
        FILE_PREFIX = (
            f"WordFluency_Animal_{timestamp}_"
            f"{SUBJECT_ID}_{SUBJECT_NAME}"
        )
    else:
        FILE_PREFIX = (
            f"WordFluency_Animal_{timestamp}_{SUBJECT_ID}"
        )

    OUTPUT_DIR = os.path.join(
        SUBJECT_DIR,
        FILE_PREFIX
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    AUDIO_FILE = os.path.join(
        OUTPUT_DIR,
        f"{FILE_PREFIX}_recorded_speech.wav"
    )

    TXT_FILE = os.path.join(
        OUTPUT_DIR,
        f"{FILE_PREFIX}_analysis_result.txt"
    )

    print("")
    print(f"被験者フォルダー: {SUBJECT_DIR}")
    print(f"出力フォルダー: {OUTPUT_DIR}")

    try:
        show_audio_devices()
    except Exception as e:
        print("")
        print("[エラー] 音声入力デバイスを確認できませんでした。")
        print(f"詳細: {e}")
        sys.exit(1)

    print("")
    print(f"Whisper {WHISPER_MODEL_NAME} モデルをロード中...")
    print(f"共有モデルフォルダー: {MODEL_DIR}")

    try:
        WHISPER_MODEL = whisper.load_model(
            WHISPER_MODEL_NAME,
            download_root=MODEL_DIR
        )
    except Exception as e:
        print("")
        print("[エラー] Whisper モデルのロードに失敗しました。")
        print(f"詳細: {e}")
        sys.exit(1)

    print("Whisper モデルのロードが完了しました。")

    print("")
    print("開始インストラクションを表示します。")

    if not show_start_instruction():
        print("課題がキャンセル（またはEscキーで終了）されました。")
        sys.exit(0)

    print("")
    print("課題画面を表示します。")
    print("画面上で Enter キーを押すと録音を開始します。")

    recorded_file = show_task_and_record_gui(
        output_filename=AUDIO_FILE,
        fs=FS,
        duration=TASK_DURATION
    )

    if recorded_file is None or not os.path.exists(recorded_file):
        print("")
        print("[キャンセル/エラー] 録音ファイルが保存されなかったため、実験を終了します。")
        sys.exit(0)

    transcription = transcribe_audio(
        audio_path=recorded_file,
        model=WHISPER_MODEL
    )

    analysis = analyze_word_fluency(transcription)

    save_analysis_to_txt(
        analysis_results=analysis,
        output_filename=TXT_FILE
    )

    print("")
    print("==================================================")
    print("実験完了")
    print("==================================================")
    print(f"音声ファイル: {AUDIO_FILE}")
    print(f"結果ファイル: {TXT_FILE}")
    print("==================================================")
    print("")