# ============================================================
# Description.py
# 情景画説明課題
# ============================================================

import csv
from collections import Counter
import datetime
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

from janome.tokenizer import Tokenizer
from PIL import Image, ImageTk


# ============================================================
# 基本設定
# ============================================================

FS = 44100

DEFAULT_WHISPER_MODEL_NAME = "turbo"
DEFAULT_MAX_DURATION = 240  # デフォルト制限時間：240秒[cite: 2]

POS_ORDER = [
    "名詞",
    "動詞",
    "形容詞",
    "形容動詞",
    "副詞",
    "連体詞",
    "接続詞",
    "助詞",
    "助動詞",
    "感動詞",
    "接頭詞",
    "フィラー",
    "その他"
]


# ============================================================
# パス設定
# ============================================================

def get_app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)

    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = get_app_dir()

SL_TOOL_DIR = os.path.dirname(APP_DIR)

IMG_FOLDER = os.path.join(
    SL_TOOL_DIR,
    "stimuli",
    "Description"
)

RESULTS_DIR = os.path.join(
    SL_TOOL_DIR,
    "results"
)

MODEL_DIR = os.path.join(
    SL_TOOL_DIR,
    "whisper_model"
)

os.environ["PATH"] = (
    SL_TOOL_DIR
    + os.pathsep
    + os.environ.get("PATH", "")
)


# ============================================================
# Whisper モデル選択 GUI (フルスクリーン・上下選択対応)
# ============================================================

def select_model():
    root = tk.Tk()

    root.title("Whisper モデル選択")
    root.configure(bg="lightgreen")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    selected_model = tk.StringVar(
        value=DEFAULT_WHISPER_MODEL_NAME
    )

    selection_result = [None]

    container = tk.Frame(root, bg="lightgreen")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【Whisper 音声認識モデルの選択】",
        font=("Meiryo", 32, "bold"),
        bg="lightgreen",
        fg="black"
    ).pack(
        pady=(0, 15)
    )

    tk.Label(
        container,
        text="文字起こしに使用するモデルを選択してください。\n（上下キーで選択変更、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 16),
        bg="lightgreen",
        fg="darkgreen",
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
            "高速処理向け（初期選択）"
        ),
        (
            "large",
            "Large",
            "高精度、処理時間・必要メモリは大きい"
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
        bg="lightgreen"
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
# 制限時間設定 GUI
# ============================================================

def select_duration():
    root = tk.Tk()

    root.title("制限時間設定")
    root.configure(bg="lightgreen")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)

    duration_var = tk.StringVar(value=str(DEFAULT_MAX_DURATION))
    result = [None]

    container = tk.Frame(root, bg="lightgreen")
    container.pack(expand=True)

    tk.Label(
        container,
        text="【課題制限時間の設定】",
        font=("Meiryo", 32, "bold"),
        bg="lightgreen",
        fg="black"
    ).pack(
        pady=(0, 15)
    )

    tk.Label(
        container,
        text="1試行あたりの制限時間（秒）を半角数字で入力してください。\n（指定時間を超えると自動的に録音が終了します）",
        font=("Meiryo", 16),
        bg="lightgreen",
        fg="darkgreen",
        justify="center"
    ).pack(
        pady=(0, 30)
    )

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

    def confirm_duration(event=None):
        try:
            val = int(duration_var.get().strip())
            if val <= 0:
                raise ValueError
            result[0] = val
            try:
                root.destroy()
            except Exception:
                pass
        except ValueError:
            messagebox.showerror("入力エラー", "制限時間は正の整数で入力してください。")

    def cancel_duration(event=None):
        result[0] = None
        try:
            root.destroy()
        except Exception:
            pass

    root.bind_all("<Return>", confirm_duration)
    root.bind_all("<Escape>", cancel_duration)

    tk.Button(
        button_frame,
        text="決定して次へ (Enter)",
        command=confirm_duration,
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

    return result[0]


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
            "【情景画説明課題】\n\n"
            "表示される絵を見て、\n"
            "できるだけ詳しく説明してください。\n\n"
            "話し終わったら Enter キーを押してください。\n"
            "（途中で終了する場合は Esc キーを押してください）\n\n"
            "準備ができたら Enter キーを押してください。"
        ),
        font=("Meiryo", 36, "bold"),
        bg="lightgreen",
        justify="center"
    ).pack(
        expand=True
    )

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

    root.after(10, root.lift)
    root.after(50, root.focus_force)
    root.after(100, root.focus_set)

    root.mainloop()

    time.sleep(0.1)

    return started[0]


# ============================================================
# 録音・画像提示（Tkinter root.after による安全な制限時間管理）
# ============================================================

def show_image_and_record_gui(
    image_path,
    output_filename,
    max_duration,
    fs=FS
):
    audio_buffer = []

    recording_active = threading.Event()
    stream_ready = threading.Event()
    stop_recording = threading.Event()

    audio_error = [None]
    aborted = [False]
    timed_out = [False]

    def audio_monitor():
        def callback(
            indata,
            frames,
            time_info,
            status
        ):
            if status:
                print(f"[音声入力警告] {status}")

            if recording_active.is_set():
                audio_buffer.append(
                    indata.copy()
                )

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
                    time.sleep(0.005)

        except Exception as e:
            audio_error[0] = e
            stream_ready.set()
            print(f"[音声入力エラー] {e}")

    audio_thread = threading.Thread(
        target=audio_monitor,
        daemon=True
    )

    audio_thread.start()

    if not stream_ready.wait(timeout=3):
        print("[警告] マイク入力ストリームの準備に時間がかかっています。")

    if audio_error[0] is not None:
        print(f"[エラー] マイク入力を開始できませんでした: {audio_error[0]}")
        return None, True, False

    root = tk.Tk()

    root.title("情景画説明課題")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="white")

    try:
        image_pil = Image.open(image_path)
        image_tk = ImageTk.PhotoImage(image_pil)

        image_label = tk.Label(
            root,
            image=image_tk,
            bg="white"
        )

        image_label.image = image_tk
        image_label.pack(
            expand=True,
            fill=tk.BOTH
        )

    except Exception as e:
        print(f"[画像エラー] {e}")
        stop_recording.set()
        try:
            root.destroy()
        except Exception:
            pass
        audio_thread.join(timeout=1)
        return None, True, False

    status_label = tk.Label(
        root,
        text=(
            "【録音中】絵をできるだけ詳しく説明してください。\n"
            "終了時は Enter キー、中断時は Esc キーを押してください。"
        ),
        font=("Meiryo", 22, "bold"),
        fg="white",
        bg="red",
        pady=18,
        justify="center"
    )

    status_label.pack(
        fill=tk.X,
        side=tk.BOTTOM
    )

    root.update_idletasks()
    root.update()

    # 録音開始
    recording_active.set()
    print("[刺激提示] 情景画を提示しました。録音を開始します。")

    def close_window():
        stop_recording.set()
        try:
            root.destroy()
        except Exception:
            pass

    def end_recording(event=None):
        if stop_recording.is_set():
            return
        print("[キー検知] 録音を終了して保存します。")
        close_window()

    def abort_task(event=None):
        if stop_recording.is_set():
            return
        print("[キー検知] 課題が中断されました。")
        aborted[0] = True
        close_window()

    def handle_timeout():
        if stop_recording.is_set():
            return
        print("[タイムアウト] 制限時間に達したため録音を終了します。")
        timed_out[0] = True
        try:
            status_label.config(
                text="【タイムアウト】制限時間に達しました。",
                fg="white",
                bg="orange"
            )
            root.update()
            time.sleep(0.5)
        except Exception:
            pass
        close_window()

    # キーバインド設定
    root.bind_all("<Return>", end_recording)
    root.bind_all("<space>", end_recording)
    root.bind_all("<Escape>", abort_task)

    # 制限時間のタイマーをセット（秒をミリ秒に変換）
    timeout_ms = int(max_duration * 1000)
    root.after(timeout_ms, handle_timeout)

    def keep_focus():
        try:
            if root.winfo_exists() and not stop_recording.is_set():
                root.lift()
                root.focus_force()
                root.focus_set()
                root.after(250, keep_focus)
        except Exception:
            pass

    root.lift()
    root.focus_force()
    root.focus_set()
    root.after(100, keep_focus)

    root.mainloop()

    # メインループ終了後の後処理
    stop_recording.set()
    audio_thread.join(timeout=2)

    if aborted[0]:
        return None, True, False

    if not audio_buffer:
        print("[エラー] 録音データが取得できませんでした。")
        return None, False, timed_out[0]

    try:
        recorded_audio = np.concatenate(
            audio_buffer,
            axis=0
        )
    except Exception as e:
        print(f"[エラー] 録音データを結合できませんでした: {e}")
        return None, False, timed_out[0]

    try:
        sf.write(
            output_filename,
            recorded_audio,
            fs
        )
    except Exception as e:
        print(f"[エラー] WAVファイルの保存に失敗しました: {e}")
        return None, False, timed_out[0]

    if not os.path.exists(output_filename):
        print(f"[エラー] WAVファイルが保存されませんでした: {output_filename}")
        return None, False, timed_out[0]

    return output_filename, False, timed_out[0]


# ============================================================
# 録音時間の取得
# ============================================================

def get_audio_duration_seconds(audio_path):
    try:
        audio_info = sf.info(audio_path)

        if audio_info.samplerate <= 0:
            return 0.0

        return round(
            audio_info.frames / audio_info.samplerate,
            3
        )

    except Exception as e:
        print(
            "[警告] 録音時間を取得できませんでした: "
            f"{e}"
        )

        return 0.0


# ============================================================
# Whisper 音声認識
# ============================================================

def transcribe_audio(
    audio_path,
    model
):
    if not os.path.exists(audio_path):
        print(
            "[エラー] 音声ファイルが見つかりません: "
            f"{audio_path}"
        )

        return {
            "text": "",
            "utterances": []
        }

    print("\n文字起こしを実行中...")

    try:
        result = model.transcribe(
            audio_path,
            language="ja",
            task="transcribe",
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.0
        )

    except Exception as e:
        print(
            "[エラー] Whisperの文字起こしに失敗しました: "
            f"{e}"
        )

        return {
            "text": "",
            "utterances": []
        }

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
        full_text = full_text.replace(
            phrase,
            ""
        )

    return {
        "text": full_text.strip(),
        "utterances": utterances
    }


# ============================================================
# 談話・品詞分析
# ============================================================

def analyze_discourse(
    transcription_result,
    recording_duration_seconds
):
    full_text = transcription_result.get(
        "text",
        ""
    ).strip()

    utterances = transcription_result.get(
        "utterances",
        []
    )

    recording_duration_minutes = (
        recording_duration_seconds / 60
        if recording_duration_seconds > 0
        else 0
    )

    empty_counts = {
        pos: 0
        for pos in POS_ORDER
    }

    empty_ratios = {
        pos: 0.0
        for pos in POS_ORDER
    }

    if not full_text:
        return {
            "全文": "（無音または認識結果なし）",
            "総発話数": 0,
            "総単語数": 0,
            "異なり単語数": 0,
            "平均発話長": 0,
            "TTR": 0,
            "Brunet": 0,
            "録音時間_s": recording_duration_seconds,
            "録音時間_min": round(
                recording_duration_minutes,
                4
            ),
            "発話速度_words_per_min": 0,
            "発話速度_words_per_sec": 0,
            "品詞別語数": empty_counts,
            "品詞別割合": empty_ratios
        }

    tokenizer = Tokenizer()

    tokens = list(
        tokenizer.tokenize(full_text)
    )

    content_tokens = [
        token
        for token in tokens
        if token.part_of_speech.split(",")[0] != "記号"
    ]

    words = [
        token.surface
        for token in content_tokens
    ]

    pos_tags = [
        token.part_of_speech.split(",")[0]
        for token in content_tokens
    ]

    base_forms = [
        token.base_form
        for token in content_tokens
    ]

    total_words = len(words)
    unique_words = len(set(base_forms))

    pos_counts = {
        pos: 0
        for pos in POS_ORDER
    }

    for pos in pos_tags:
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            pos_counts["その他"] += 1

    pos_ratios = {
        pos: round(
            (count / total_words) * 100,
            1
        )
        if total_words > 0
        else 0.0
        for pos, count in pos_counts.items()
    }

    total_utterances = (
        len(utterances)
        if utterances
        else 1
    )

    average_utterance_length = (
        total_words / total_utterances
        if total_utterances > 0
        else 0
    )

    ttr = (
        unique_words / total_words
        if total_words > 0
        else 0
    )

    brunet = (
        total_words ** (unique_words ** -0.165)
        if total_words > 0 and unique_words > 0
        else 0
    )

    words_per_min = (
        total_words / recording_duration_minutes
        if recording_duration_minutes > 0
        else 0
    )

    words_per_sec = (
        total_words / recording_duration_seconds
        if recording_duration_seconds > 0
        else 0
    )

    return {
        "全文": full_text,
        "総発話数": total_utterances,
        "総単語数": total_words,
        "異なり単語数": unique_words,
        "平均発話長": round(
            average_utterance_length,
            2
        ),
        "TTR": round(ttr, 4),
        "Brunet": round(brunet, 4),
        "録音時間_s": round(
            recording_duration_seconds,
            3
        ),
        "録音時間_min": round(
            recording_duration_minutes,
            4
        ),
        "発話速度_words_per_min": round(
            words_per_min,
            2
        ),
        "発話速度_words_per_sec": round(
            words_per_sec,
            3
        ),
        "品詞別語数": pos_counts,
        "品詞別割合": pos_ratios
    }


# ============================================================
# 画像一覧取得
# ============================================================

def get_description_images(folder_path):
    image_files = []

    if not os.path.exists(folder_path):
        return image_files

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    for filename in os.listdir(folder_path):
        name, extension = os.path.splitext(filename)

        if extension.lower() not in valid_extensions:
            continue

        if "_" not in name:
            continue

        number_part, stimulus_name = name.split(
            "_",
            1
        )

        if not number_part.isdigit():
            continue

        image_files.append(
            (
                int(number_part),
                filename,
                stimulus_name
            )
        )

    image_files.sort(
        key=lambda item: item[0]
    )

    return image_files


# ============================================================
# TXT 保存
# ============================================================

def save_trial_txt(
    output_filename,
    subject_id,
    subject_name,
    whisper_model_name,
    max_duration,
    result
):
    analysis = result["analysis"]

    with open(
        output_filename,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "==================================================\n"
        )
        f.write(
            "情景画説明課題：談話分析結果\n"
        )
        f.write(
            "==================================================\n\n"
        )

        f.write(f"被験者ID: {subject_id}\n")
        f.write(f"被験者名: {subject_name}\n")
        f.write(
            f"Whisperモデル: {whisper_model_name}\n"
        )
        f.write(
            f"最大制限時間_s: {max_duration}\n"
        )
        f.write(
            f"試行番号: {result['trial']}\n"
        )
        f.write(
            f"刺激画像: {result['image']}\n"
        )
        f.write(
            f"タイムアウト判定: {'はい' if result['timed_out'] else 'いいえ'}\n"
        )
        f.write(
            "被験者録音名: "
            f"{os.path.basename(result['audio'])}\n\n"
        )

        for key in [
            "全文",
            "総発話数",
            "総単語数",
            "異なり単語数",
            "平均発話長",
            "TTR",
            "Brunet",
            "録音時間_s",
            "録音時間_min",
            "発話速度_words_per_min",
            "発話速度_words_per_sec"
        ]:
            f.write(
                f"{key}: {analysis.get(key, '')}\n"
            )

        f.write("\n品詞別語数・割合:\n")

        for pos in POS_ORDER:
            f.write(
                f"{pos}: "
                f"{analysis['品詞別語数'].get(pos, 0)}語 / "
                f"{analysis['品詞別割合'].get(pos, 0.0)}%\n"
            )


# ============================================================
# 統合 CSV 保存
# ============================================================

def save_total_results_csv(
    all_results,
    output_filename,
    subject_id,
    subject_name,
    whisper_model_name,
    max_duration
):
    measurement_time = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    headers = [
        "被験者ID",
        "被験者名",
        "測定日時",
        "Whisperモデル",
        "最大制限時間_s",
        "試行番号",
        "刺激名",
        "刺激画像ファイル",
        "タイムアウト",
        "被験者録音名",
        "全文",
        "総発話数",
        "総単語数",
        "異なり単語数",
        "平均発話長",
        "TTR",
        "Brunet",
        "録音時間_s",
        "録音時間_min",
        "発話速度_words_per_min",
        "発話速度_words_per_sec"
    ]

    for pos in POS_ORDER:
        headers.append(
            f"{pos}_語数"
        )

    for pos in POS_ORDER:
        headers.append(
            f"{pos}_割合(%)"
        )

    with open(
        output_filename,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.writer(
            f,
            lineterminator="\n"
        )

        writer.writerow(headers)

        for result in all_results:
            analysis = result["analysis"]

            image_filename = result["image"]

            row = [
                subject_id,
                subject_name,
                measurement_time,
                whisper_model_name,
                max_duration,
                result["trial"],
                os.path.splitext(image_filename)[0],
                image_filename,
                "はい" if result["timed_out"] else "いいえ",
                os.path.basename(result["audio"]),
                analysis["全文"],
                analysis["総発話数"],
                analysis["総単語数"],
                analysis["異なり単語数"],
                analysis["平均発話長"],
                analysis["TTR"],
                analysis["Brunet"],
                analysis["録音時間_s"],
                analysis["録音時間_min"],
                analysis["発話速度_words_per_min"],
                analysis["発話速度_words_per_sec"]
            ]

            for pos in POS_ORDER:
                row.append(
                    analysis["品詞別語数"].get(
                        pos,
                        0
                    )
                )

            for pos in POS_ORDER:
                row.append(
                    analysis["品詞別割合"].get(
                        pos,
                        0.0
                    )
                )

            writer.writerow(row)


# ============================================================
# メイン処理
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "使用方法:\n"
            "python Description.py [被験者ID] [被験者名]\n\n"
            "例:\n"
            "python Description.py 001 山田太郎"
        )

        sys.exit(1)

    SUBJECT_ID = sys.argv[1]

    SUBJECT_NAME = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else ""
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # 1. Whisperモデル選択
    WHISPER_MODEL_NAME = select_model()

    if WHISPER_MODEL_NAME is None:
        print("モデル選択がキャンセル（またはEscキーで終了）されました。")
        sys.exit(0)

    # 2. 制限時間設定
    MAX_DURATION = select_duration()

    if MAX_DURATION is None:
        print("制限時間設定がキャンセル（またはEscキーで終了）されました。")
        sys.exit(0)

    image_files = get_description_images(
        IMG_FOLDER
    )

    if not image_files:
        print(
            "[エラー] 情景画が見つかりませんでした。\n"
            f"{IMG_FOLDER}"
        )

        sys.exit(1)

    print(
        f"Whisper {WHISPER_MODEL_NAME} モデルをロード中..."
    )

    try:
        WHISPER_MODEL = whisper.load_model(
            WHISPER_MODEL_NAME,
            download_root=MODEL_DIR
        )

    except Exception as e:
        print(
            "[エラー] Whisperモデルのロードに失敗しました。\n"
            f"{e}"
        )

        sys.exit(1)

    if not show_start_instruction():
        print("課題がキャンセル（またはEscキーで終了）されました。")
        sys.exit(0)

    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    if SUBJECT_NAME:
        subject_folder = (
            f"{SUBJECT_ID}_{SUBJECT_NAME}"
        )

        file_prefix = (
            f"Description_{timestamp}_"
            f"{SUBJECT_ID}_{SUBJECT_NAME}"
        )

    else:
        subject_folder = SUBJECT_ID

        file_prefix = (
            f"Description_{timestamp}_{SUBJECT_ID}"
        )

    subject_dir = os.path.join(
        RESULTS_DIR,
        subject_folder
    )

    output_dir = os.path.join(
        subject_dir,
        file_prefix
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    all_results = []
    task_aborted = False

    for index, (
        trial_number,
        image_filename,
        stimulus_name
    ) in enumerate(
        image_files,
        start=1
    ):
        print("")
        print("=" * 60)
        print(f"試行 {index}/{len(image_files)}")
        print(f"刺激名: {stimulus_name}")
        print("=" * 60)

        image_path = os.path.join(
            IMG_FOLDER,
            image_filename
        )

        audio_file = os.path.join(
            output_dir,
            f"{file_prefix}_trial{trial_number:02d}_"
            "recorded_speech.wav"
        )

        txt_file = os.path.join(
            output_dir,
            f"{file_prefix}_trial{trial_number:02d}_"
            "analysis_result.txt"
        )

        recorded_file, aborted, timed_out = show_image_and_record_gui(
            image_path=image_path,
            output_filename=audio_file,
            max_duration=MAX_DURATION,
            fs=FS
        )

        if aborted:
            print("\n[中断] ユーザー操作（Escapeキー）により課題が中断されました。")
            task_aborted = True
            break

        if recorded_file is None:
            print(
                "[エラー] 録音に失敗したため、"
                "この試行をスキップします。"
            )
            continue

        transcription = transcribe_audio(
            audio_path=recorded_file,
            model=WHISPER_MODEL
        )

        recording_duration_seconds = (
            get_audio_duration_seconds(
                recorded_file
            )
        )

        analysis = analyze_discourse(
            transcription_result=transcription,
            recording_duration_seconds=(
                recording_duration_seconds
            )
        )

        result = {
            "trial": trial_number,
            "image": image_filename,
            "audio": audio_file,
            "timed_out": timed_out,
            "analysis": analysis
        }

        save_trial_txt(
            output_filename=txt_file,
            subject_id=SUBJECT_ID,
            subject_name=SUBJECT_NAME,
            whisper_model_name=WHISPER_MODEL_NAME,
            max_duration=MAX_DURATION,
            result=result
        )

        all_results.append(result)

        print(
            f"録音時間_s: {analysis['録音時間_s']}"
        )

        print(
            f"録音時間_min: {analysis['録音時間_min']}"
        )

        print(
            "発話速度_words_per_min: "
            f"{analysis['発話速度_words_per_min']}"
        )

        print(
            "発話速度_words_per_sec: "
            f"{analysis['発話速度_words_per_sec']}"
        )

    if task_aborted:
        print("\n[中断] 課題が途中で終了されました（中断までに完了した試行のみ保存されます）。")

    total_csv_file = os.path.join(
        output_dir,
        f"{file_prefix}_total.csv"
    )

    save_total_results_csv(
        all_results=all_results,
        output_filename=total_csv_file,
        subject_id=SUBJECT_ID,
        subject_name=SUBJECT_NAME,
        whisper_model_name=WHISPER_MODEL_NAME,
        max_duration=MAX_DURATION
    )

    print("")
    print("=" * 60)
    print("実験完了" if not task_aborted else "実験中断")
    print("=" * 60)
    print(f"完了試行数: {len(all_results)}")
    print(f"出力フォルダー: {output_dir}")
    print(f"統合CSV: {total_csv_file}")
    print("=" * 60)