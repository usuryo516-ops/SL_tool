import datetime
import os
import sys
import time
import tkinter as tk
import ctypes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import find_peaks


# ============================================================
# 基本設定
# ============================================================

FS = 44100
DEFAULT_DURATION = 5  # ディアドココキネシスの標準測定時間（5秒）

# 4種類の連続実施課題の定義
TASK_OPTIONS = [
    {"name": "Pa", "label": "ぱ・ぱ・ぱ（口唇音）"},
    {"name": "Ta", "label": "た・た・た（歯茎音）"},
    {"name": "Ka", "label": "か・か・か（軟口蓋音）"},
    {"name": "Pataka", "label": "ぱ・た・か・ぱ・た・か（連続運動）"},
]


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

os.environ["PATH"] = SL_TOOL_DIR + os.pathsep + os.environ.get("PATH", "")


# ============================================================
# 強制フォーカス処理
# ============================================================

def focus_window(r):
    try:
        r.deiconify()
        r.lift()
        r.attributes('-topmost', True)
        r.update_idletasks()
        h = r.winfo_id()
        u = ctypes.windll.user32
        u.ShowWindow(h, 9)
        u.BringWindowToTop(h)
        u.SetForegroundWindow(h)
        r.focus_force()
        r.focus_set()
    except Exception:
        pass


# ============================================================
# 開始・継続インストラクション
# ============================================================

def show_start_instruction(task_info, current_num, total_num):
    root = tk.Tk()
    root.title(f"ディアドココキネシス ({current_num}/{total_num})")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.configure(bg="lightblue")

    instruction_text = (
        f"【ディアドココキネシス 課題 {current_num} / {total_num}】\n\n"
        f"「 {task_info['label']} 」を\n"
        f"できるだけ速く、均等に繰り返してください。\n"
        f"（測定時間: {DEFAULT_DURATION}秒間）\n\n"
        "準備ができたら Enter キーを押してください\n"
        "（Escキーで実験終了）"
    )

    tk.Label(
        root,
        text=instruction_text,
        font=("Meiryo", 36, "bold"),
        bg="lightblue",
        fg="black",
        justify="center"
    ).pack(expand=True)

    started = [False]

    def start_task(event=None):
        started[0] = True
        try:
            root.destroy()
        except Exception:
            pass
        return "break"

    def cancel_task(event=None):
        print("\n[中断] Escキーが押されたため、実験を終了します。")
        try:
            root.destroy()
        except Exception:
            pass
        sys.exit(0)

    root.bind_all("<Return>", start_task)
    root.bind_all("<KP_Enter>", start_task)
    root.bind_all("<space>", start_task)
    root.bind_all("<Escape>", cancel_task)

    focus_window(root)
    root.after(100, lambda: focus_window(root))
    root.mainloop()

    time.sleep(0.1)
    return started[0]


# ============================================================
# 録音画面（5秒間カウントダウン付き）
# ============================================================

def show_task_and_record_gui(output_filename, task_info, current_num, total_num, fs=FS, duration=DEFAULT_DURATION):
    root = tk.Tk()
    root.title(f"ディアドココキネシス ({current_num}/{total_num}) - {task_info['label']}")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="white")

    tk.Label(
        root,
        text=f"【 課題 {current_num}/{total_num}: {task_info['label']} 】\nできるだけ速く繰り返してください",
        font=("Meiryo", 42, "bold"),
        fg="black",
        bg="white",
        justify="center"
    ).pack(expand=True, pady=(50, 20))

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
        text="【準備中】Enter キーで開始 / Esc キーで実験終了",
        font=("Meiryo", 28, "bold"),
        fg="blue",
        bg="lightgreen",
        pady=20
    )
    status_label.pack(fill=tk.X, side=tk.BOTTOM)

    recorded_audio = [None]
    timer_value = [duration]
    recording_started = [False]

    def update_timer():
        if not recording_started[0]:
            return

        if timer_value[0] > 0:
            timer_value[0] -= 1
            timer_label.config(text=f"残り時間: {timer_value[0]} 秒")
            root.after(1000, update_timer)
        else:
            timer_label.config(text="終了！", fg="green", bg="lightgreen")
            status_label.config(text="【終了】解析中...", fg="gray", bg="lightgray")
            root.after(500, root.quit)

    def start_recording_and_timer(event=None):
        if recording_started[0]:
            return

        recording_started[0] = True
        status_label.config(text="【録音中】発話してください！", fg="white", bg="red")

        try:
            recorded_audio[0] = sd.rec(
                int(duration * fs),
                samplerate=fs,
                channels=1,
                dtype="float32"
            )
        except Exception as e:
            print(f"[録音開始エラー] {e}")
            root.quit()
            return

        root.after(1000, update_timer)
        return "break"

    def cancel_task(event=None):
        print("\n[中断] Escキーが押されたため、実験を終了します。")
        try:
            sd.stop()
        except Exception:
            pass
        try:
            root.quit()
        except Exception:
            pass
        sys.exit(0)

    root.bind_all("<Return>", start_recording_and_timer)
    root.bind_all("<KP_Enter>", start_recording_and_timer)
    root.bind_all("<space>", start_recording_and_timer)
    root.bind_all("<Escape>", cancel_task)

    focus_window(root)
    root.after(100, lambda: focus_window(root))
    root.mainloop()

    try:
        root.destroy()
    except Exception:
        pass

    if recorded_audio[0] is None:
        return None

    try:
        sd.wait()
    except Exception:
        pass

    sf.write(output_filename, recorded_audio[0], fs)
    return output_filename


# ============================================================
# 波形 ＋ スペクトログラム（ピーク位置付き）の描画・保存
# ============================================================

def plot_waveform_and_spectrogram(
    full_data,
    fs,
    task_info,
    output_image_path,
    peaks_full=None,
    envelope_full=None,
    threshold=None,
    trim_range=None
):
    try:
        time_axis = np.arange(len(full_data)) / fs

        fig, (ax_wave, ax_spec) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
        )

        # 上段：時間波形・包絡線・検出ピーク
        ax_wave.plot(time_axis, full_data, color="#3b6e8c", linewidth=0.7, alpha=0.85, label="Waveform")
        if envelope_full is not None:
            ax_wave.plot(time_axis, envelope_full, color="#e07a22", linewidth=1.3, label="Envelope")
        if peaks_full is not None and len(peaks_full) > 0:
            ax_wave.scatter(
                time_axis[peaks_full],
                envelope_full[peaks_full] if envelope_full is not None else full_data[peaks_full],
                color="#d62828",
                s=28,
                zorder=5,
                label=f"Detected Peaks ({len(peaks_full)})"
            )
        if threshold is not None:
            ax_wave.axhline(threshold, color="#888888", linestyle="--", linewidth=0.9, label="Threshold")
        if trim_range is not None:
            t_start, t_end = trim_range
            ax_wave.axvline(t_start, color="#2a9d8f", linestyle=":", linewidth=1.2)
            ax_wave.axvline(t_end, color="#2a9d8f", linestyle=":", linewidth=1.2)

        task_label_en = task_info.get("name", "")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_title(f"DDK Waveform & Spectrogram - {task_label_en}")
        ax_wave.grid(True, linestyle="--", alpha=0.35)
        ax_wave.legend(loc="upper right", fontsize=8)

        # 下段：スペクトログラム（広帯域で音節の立ち上がりを把握しやすい設定）
        ax_spec.specgram(
            full_data,
            Fs=fs,
            NFFT=512,
            noverlap=384,
            cmap="magma",
            vmin=-90,
            vmax=-10
        )
        if peaks_full is not None and len(peaks_full) > 0:
            for p_idx in peaks_full:
                ax_spec.axvline(time_axis[p_idx], color="#4cc9f0", linestyle="--", linewidth=0.7, alpha=0.75)

        ax_spec.set_ylim(0, 8000)
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        plt.savefig(output_image_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return os.path.basename(output_image_path)
    except Exception as e:
        print(f"[警告] 波形・スペクトログラム描画エラー: {e}")
        return ""


# ============================================================
# 音響ピーク解析による正確な構音回数カウント
# ============================================================

def analyze_diadochokinesis_acoustic(audio_path, task_info, duration=DEFAULT_DURATION, plot_output_path=None):
    audio_path = os.path.normpath(audio_path)
    if not os.path.exists(audio_path):
        return {
            "課題名": task_info["label"],
            "測定時間(秒)": duration,
            "総音節回数": 0,
            "秒間音節回数(回/秒)": 0.0,
            "ぱたかセット換算": "-",
            "波形_スペクトログラム画像": ""
        }

    data, fs = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    window_size = max(1, int(fs * 0.015))  # 15msの平滑化
    full_envelope = np.convolve(np.abs(data), np.ones(window_size) / window_size, mode="same")

    # 前後の無音をカット
    abs_data_full = np.abs(data)
    max_amp_full = np.max(abs_data_full)
    if max_amp_full < 0.005:
        img_name = ""
        if plot_output_path:
            img_name = plot_waveform_and_spectrogram(
                full_data=data,
                fs=fs,
                task_info=task_info,
                output_image_path=plot_output_path,
                peaks_full=np.array([], dtype=int),
                envelope_full=full_envelope
            )
        return {
            "課題名": task_info["label"],
            "測定時間(秒)": duration,
            "総音節回数": 0,
            "秒間音節回数(回/秒)": 0.0,
            "ぱたかセット換算": "-",
            "波形_スペクトログラム画像": img_name
        }

    threshold_trim = max_amp_full * 0.05
    active_idx = np.where(abs_data_full > threshold_trim)[0]
    if len(active_idx) == 0:
        start, end = 0, len(data)
        trimmed_data = data
    else:
        pad = int(fs * 0.05)
        start = max(0, active_idx[0] - pad)
        end = min(len(data), active_idx[-1] + pad)
        trimmed_data = data[start:end]

    actual_duration = len(trimmed_data) / fs
    calc_duration = max(actual_duration, 1.0)

    # 包絡線（エンベロープ）を作成して破裂音のピークを検出
    abs_data = np.abs(trimmed_data)
    envelope = np.convolve(abs_data, np.ones(window_size) / window_size, mode="same")

    max_amp = np.max(envelope)
    noise_floor = np.percentile(envelope, 20)
    threshold = max(max_amp * 0.12, noise_floor * 2.5)

    # 人間の発話速度の限界（1秒間に最大約10〜11回＝最小間隔0.07秒）
    min_distance = int(fs * 0.07)

    peaks, _ = find_peaks(envelope, height=threshold, distance=min_distance, prominence=threshold * 0.4)
    peaks_full = peaks + start
    syllable_count = len(peaks)

    sps = round(syllable_count / calc_duration, 2)

    if task_info["name"] == "Pataka":
        pataka_sets = round(syllable_count / 3.0, 2)
        pataka_sps = round(pataka_sets / calc_duration, 2)
        set_info = f"{pataka_sets} セット ({pataka_sps} セット/秒)"
    else:
        set_info = "-"

    img_name = ""
    if plot_output_path:
        img_name = plot_waveform_and_spectrogram(
            full_data=data,
            fs=fs,
            task_info=task_info,
            output_image_path=plot_output_path,
            peaks_full=peaks_full,
            envelope_full=full_envelope,
            threshold=threshold,
            trim_range=(start / fs, end / fs)
        )

    print(f"\n--- 音響解析結果 ({task_info['label']}) ---")
    print(f"-> 総音節回数: {syllable_count} 回 | 秒間回数: {sps} 回/秒 (発話実時間: {actual_duration:.2f}秒)")
    if task_info["name"] == "Pataka":
        print(f"-> 「ぱたか」セット換算: {set_info}")
    if img_name:
        print(f"-> 波形・スペクトログラム保存: {img_name}")

    return {
        "課題名": task_info["label"],
        "測定時間(秒)": duration,
        "総音節回数": syllable_count,
        "秒間音節回数(回/秒)": sps,
        "ぱたかセット換算": set_info,
        "波形_スペクトログラム画像": img_name
    }


# ============================================================
# 結果保存
# ============================================================

def save_all_results_to_txt(all_results, output_filename):
    output_filename = os.path.normpath(output_filename)
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("       ディアドココキネシス 総合評価結果（音響ピーク解析）\n")
            f.write("==================================================\n\n")

            for idx, res in enumerate(all_results, start=1):
                f.write(f"--- 課題 {idx}: {res['課題名']} ---\n")
                f.write(f"■ 測定時間: {res['測定時間(秒)']} 秒\n")
                f.write(f"■ 総音節回数: {res['総音節回数']} 回\n")
                f.write(f"■ 秒間音節回数: {res['秒間音節回数(回/秒)']} 回/秒\n")
                if res['ぱたかセット換算'] != "-":
                    f.write(f"■ 「ぱたか」3音節セット換算: {res['ぱたかセット換算']}\n")
                if res.get("波形_スペクトログラム画像"):
                    f.write(f"■ 波形・スペクトログラム画像: {res['波形_スペクトログラム画像']}\n")
                f.write("\n")

        print(f"[保存完了] 総合結果: {output_filename}")
    except Exception as e:
        print(f"[エラー] 保存失敗: {e}")


# ============================================================
# メイン処理
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("python Diadochokinesis.py [被験者ID] [被験者名]\n")
        sys.exit(1)

    SUBJECT_ID = sys.argv[1]
    SUBJECT_NAME = sys.argv[2] if len(sys.argv) >= 3 else ""

    print("\n==================================================")
    print("ディアドココキネシス評価ツール（音響ピーク解析版）")
    print("==================================================")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    subject_folder = f"{SUBJECT_ID}_{SUBJECT_NAME}" if SUBJECT_NAME else SUBJECT_ID
    SUBJECT_DIR = os.path.join(RESULTS_DIR, subject_folder)
    os.makedirs(SUBJECT_DIR, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    DATE_DIR = os.path.join(SUBJECT_DIR, date_str)
    os.makedirs(DATE_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_tag = f"Diadochokinesis_All_{timestamp}_{subject_folder}"

    OUTPUT_DIR = os.path.join(DATE_DIR, file_tag)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TXT_FILE = os.path.join(OUTPUT_DIR, f"{file_tag}_results.txt")

    print(f"\n結果保存先: {OUTPUT_DIR}")

    all_results = []
    total_tasks = len(TASK_OPTIONS)

    for idx, task_info in enumerate(TASK_OPTIONS, start=1):
        print(f"\n>>> 課題 {idx}/{total_tasks}: {task_info['label']} を開始します")

        if not show_start_instruction(task_info, idx, total_tasks):
            break

        audio_file_path = os.path.join(OUTPUT_DIR, f"task_{idx}_{task_info['name']}_recorded.wav")
        plot_image_path = os.path.join(OUTPUT_DIR, f"task_{idx}_{task_info['name']}_spectrogram.png")

        recorded_file = show_task_and_record_gui(
            output_filename=audio_file_path,
            task_info=task_info,
            current_num=idx,
            total_num=total_tasks,
            fs=FS,
            duration=DEFAULT_DURATION
        )

        if recorded_file is None or not os.path.exists(recorded_file):
            continue

        analysis = analyze_diadochokinesis_acoustic(
            audio_path=recorded_file,
            task_info=task_info,
            duration=DEFAULT_DURATION,
            plot_output_path=plot_image_path
        )

        all_results.append(analysis)
        time.sleep(0.5)

    if all_results:
        save_all_results_to_txt(all_results, TXT_FILE)

    print("\n==================================================")
    print("すべての測定・評価が完了しました")
    print("==================================================")
    print(f"結果ファイル: {TXT_FILE}")
    print("==================================================\n")