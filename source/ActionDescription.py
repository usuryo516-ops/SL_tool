import csv, ctypes, datetime, os, sys, threading, time, winsound, subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
import whisper
from janome.tokenizer import Tokenizer
from PIL import Image, ImageTk

FS = 44100
THRESHOLD = 0.08
MAX_DURATION = 15  # デフォルト値（設定画面で変更可能になります）
DEFAULT_WHISPER_MODEL_NAME = 'turbo'
CLICK_DURATION = 0.03
CLICK_FREQUENCY = 1000
CLICK_VOLUME = 0.70
WHISPER_MODEL = None

# 実験全体の中断フラグ
TASK_ABORTED = threading.Event()

def appdir(): 
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

APP_DIR = appdir()
SL_TOOL_DIR = os.path.dirname(APP_DIR)
RESULTS_DIR = os.path.join(SL_TOOL_DIR, 'results')
IMG_FOLDER = os.path.join(SL_TOOL_DIR, 'stimuli', 'ActionDescription')
MODEL_DIR = os.path.join(SL_TOOL_DIR, 'whisper_model')
os.environ['PATH'] = SL_TOOL_DIR + os.pathsep + os.environ.get('PATH', '')

# ============================================================
# フォーカス処理
# ============================================================

def focus(r):
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
# 1. Whisperモデル選択画面（上下配置・上下キー対応）
# ============================================================
def select_whisper_model():
    root = tk.Tk()
    root.title("Whisperモデル選択")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="lightblue")

    selected_model = tk.StringVar(value=DEFAULT_WHISPER_MODEL_NAME)
    result = [None]
    model_names = ["small", "turbo", "large"]

    tk.Label(
        root,
        text="音声認識モデルを選択してください",
        font=("Meiryo", 36, "bold"),
        bg="lightblue"
    ).pack(pady=(60, 20))

    tk.Label(
        root,
        text="（↑↓キーで選択変更、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 18),
        bg="lightblue",
        fg="darkblue"
    ).pack(pady=(0, 30))

    model_frame = tk.Frame(root, bg="white", padx=50, pady=30)
    model_frame.pack(pady=10)

    model_options = [
        ("small", "Small (標準モード：比較的高速で実用的な認識精度)"),
        ("turbo", "Turbo (高速モード：Enter後の待機時間を短くしたい場合)"),
        ("large", "Large (高精度モード：待ち時間が長くても精度を優先)")
    ]

    for model_name, label_text in model_options:
        option_frame = tk.Frame(model_frame, bg="white")
        option_frame.pack(fill="x", pady=10)

        tk.Radiobutton(
            option_frame,
            text=label_text,
            variable=selected_model,
            value=model_name,
            font=("Meiryo", 20),
            bg="white",
            activebackground="white",
            anchor="w"
        ).pack(anchor="w", padx=20, pady=5)

    def start_task(event=None):
        result[0] = selected_model.get()
        try: root.destroy()
        except Exception: pass
        return "break"

    def cancel_task(event=None):
        TASK_ABORTED.set()
        result[0] = None
        try: root.destroy()
        except Exception: pass
        return "break"

    def move_model_selection(direction):
        current_model = selected_model.get()
        try:
            current_index = model_names.index(current_model)
        except ValueError:
            current_index = 0
        new_index = (current_index + direction) % len(model_names)
        selected_model.set(model_names[new_index])
        return "break"

    tk.Button(
        root,
        text="決定して次へ",
        font=("Meiryo", 24, "bold"),
        bg="white",
        command=start_task,
        padx=50,
        pady=10
    ).pack(pady=30)

    root.bind_all("<Return>", start_task)
    root.bind_all("<KP_Enter>", start_task)
    root.bind_all("<Escape>", cancel_task)
    root.bind_all("<Up>", lambda event: move_model_selection(-1))
    root.bind_all("<Down>", lambda event: move_model_selection(1))

    focus(root)
    root.after(100, lambda: focus(root))
    root.mainloop()

    return result[0]

# ============================================================
# 2. ヒント有無および制限時間設定画面
# ============================================================
def select_hint_settings():
    root = tk.Tk()
    root.title("ヒント・制限時間設定")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="lightblue")

    has_hint = tk.BooleanVar(value=True)
    time_var = tk.StringVar(value=str(MAX_DURATION))
    result = [None]
    hint_values = [True, False]

    tk.Label(
        root,
        text="ヒント機能および制限時間の設定",
        font=("Meiryo", 36, "bold"),
        bg="lightblue"
    ).pack(pady=(40, 10))

    tk.Label(
        root,
        text="（↑↓キーでヒント選択、Enterキーで決定 / Escキーで終了）",
        font=("Meiryo", 18),
        bg="lightblue",
        fg="darkblue"
    ).pack(pady=(0, 20))

    settings_frame = tk.Frame(root, bg="white", padx=50, pady=25)
    settings_frame.pack(pady=10)

    tk.Label(
        settings_frame,
        text="■ ヒント機能の選択",
        font=("Meiryo", 20, "bold"),
        bg="white",
        anchor="w"
    ).pack(anchor="w", padx=20, pady=(0, 10))

    hint_options = [
        (True, "ヒントあり (1回目不正解時に聴覚ヒントを提示し再録音を実施)"),
        (False, "ヒントなし (各試行1回のみの回答で完了)")
    ]

    for val, label_text in hint_options:
        tk.Radiobutton(
            settings_frame,
            text=label_text,
            variable=has_hint,
            value=val,
            font=("Meiryo", 18),
            bg="white",
            activebackground="white",
            anchor="w"
        ).pack(anchor="w", padx=40, pady=5)

    tk.Frame(settings_frame, height=2, bg="lightgray").pack(fill="x", pady=20)

    tk.Label(
        settings_frame,
        text="■ 課題制限時間（秒）",
        font=("Meiryo", 20, "bold"),
        bg="white",
        anchor="w"
    ).pack(anchor="w", padx=20, pady=(0, 5))

    time_entry_frame = tk.Frame(settings_frame, bg="white")
    time_entry_frame.pack(anchor="w", padx=40, pady=5)

    tk.Entry(
        time_entry_frame,
        textvariable=time_var,
        font=("Meiryo", 20),
        width=6,
        justify="center"
    ).pack(side=tk.LEFT, padx=(0, 10))

    tk.Label(
        time_entry_frame,
        text="秒（半角数字で入力してください）",
        font=("Meiryo", 18),
        bg="white"
    ).pack(side=tk.LEFT)

    def start_task(event=None):
        try:
            sec = int(time_var.get().strip())
            if sec <= 0:
                raise ValueError
        except ValueError:
            tk.messagebox.showerror("入力エラー", "制限時間は正の整数で入力してください。")
            return "break"

        result[0] = (has_hint.get(), sec)
        try: root.destroy()
        except Exception: pass
        return "break"

    def cancel_task(event=None):
        TASK_ABORTED.set()
        result[0] = None
        try: root.destroy()
        except Exception: pass
        return "break"

    def move_hint_selection(direction):
        current_val = has_hint.get()
        try:
            current_index = hint_values.index(current_val)
        except ValueError:
            current_index = 0
        new_index = (current_index + direction) % len(hint_values)
        has_hint.set(hint_values[new_index])
        return "break"

    tk.Button(
        root,
        text="課題を開始",
        font=("Meiryo", 24, "bold"),
        bg="white",
        command=start_task,
        padx=50,
        pady=10
    ).pack(pady=20)

    root.bind_all("<Return>", start_task)
    root.bind_all("<KP_Enter>", start_task)
    root.bind_all("<Escape>", cancel_task)
    root.bind_all("<Up>", lambda event: move_hint_selection(-1))
    root.bind_all("<Down>", lambda event: move_hint_selection(1))

    focus(root)
    root.after(100, lambda: focus(root))
    root.mainloop()

    return result[0]

def instruction(text):
    if TASK_ABORTED.is_set():
        return False
    r = tk.Tk()
    r.title('インストラクション')
    r.attributes('-fullscreen', True)
    r.attributes('-topmost', True)
    r.configure(bg='lightblue')
    closed = [False]

    tk.Label(r, text=text, font=('Meiryo', 42), bg='lightblue', justify='center').pack(expand=True)

    def close(e=None): 
        closed[0] = True
        r.destroy()
        return 'break'

    def abort_inst(e=None):
        TASK_ABORTED.set()
        closed[0] = False
        r.destroy()
        return 'break'

    r.bind_all('<Return>', close)
    r.bind_all('<KP_Enter>', close)
    r.bind_all('<space>', close)
    r.bind_all('<Escape>', abort_inst)

    focus(r)
    r.after(100, lambda: focus(r))
    r.mainloop()
    return closed[0] and not TASK_ABORTED.is_set()

def kana(s): 
    return ''.join(chr(ord(c)-96) if 0x30a1 <= ord(c) <= 0x30f6 else c for c in s.strip()).replace('ー', '')

def action_verb(s):
    try:
        ts = list(Tokenizer().tokenize(s.strip()))
        main_verbs = []
        for t in ts:
            pos = t.part_of_speech.split(',')
            major = pos[0]
            minor = pos[1] if len(pos) > 1 else ''
            if major == '動詞' and minor != '非自立':
                main_verbs.append(t.base_form)
        
        if main_verbs:
            return main_verbs[-1]
        
        vs = [t for t in ts if t.part_of_speech.split(',')[0] == '動詞']
        return vs[-1].base_form if vs else (ts[-1].base_form if ts else s.strip())
    except Exception: 
        return s.strip()

def verb_reading(v):
    try:
        ts = list(Tokenizer().tokenize(v))
        t = next((x for x in ts if x.part_of_speech.split(',')[0] == '動詞'), ts[0] if ts else None)
        return kana(t.reading if t else v)
    except Exception: 
        return kana(v)

def pre_verb_hint(s):
    try:
        s = s.strip().rstrip('.').rstrip('。')
        tokens = list(Tokenizer().tokenize(s))
        target_verb = action_verb(s)
        last_verb_idx = -1
        
        for idx, t in enumerate(tokens):
            if t.base_form == target_verb or t.surface == target_verb:
                last_verb_idx = idx
                break
                
        target_tokens = tokens if last_verb_idx == -1 else tokens[:last_verb_idx]
            
        hint_parts = []
        current = []
        
        for t in target_tokens:
            pos = t.part_of_speech.split(',')
            major = pos[0]
            minor = pos[1] if len(pos) > 1 else ''
            
            reading = t.reading if (t.reading and t.reading != '*') else t.surface
            y = kana(reading)
            current.append(y)
            
            if major == '助詞' and minor in ['格助詞', '副助詞', '係助詞']:
                hint_parts.append(''.join(current))
                current = []
                
        if current:
            hint_parts.append(''.join(current))
            
        hint_text = ' '.join(x for x in hint_parts if x)
        return hint_text if hint_text else kana(s)
        
    except Exception as e: 
        print('[ヒント生成エラー]', e)
        return s.strip()

def correct(text, target, answer):
    if not text.strip() or not target.strip(): 
        return False
    
    negative_keywords = ['ない', 'なかった', 'ません', 'ませんでした', 'ず', 'ぬ', 'ん']
    text_has_negation = any(neg in text for neg in negative_keywords)
    answer_has_negation = any(neg in answer for neg in negative_keywords)
    
    if not answer_has_negation and text_has_negation:
        return False

    b = kana(target)
    rd = verb_reading(target)
    try:
        for t in Tokenizer().tokenize(text.strip()):
            x = [kana(t.surface), kana(t.base_form), kana(t.reading)]
            if t.base_form == target or b in x or (rd and rd in x): 
                return True
        return False
    except Exception:
        x = kana(text)
        return b in x or rd in x

def tts(text, path):
    q = lambda x: os.path.abspath(x).replace("'", "''")
    ps = f"Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;$s.SetOutputToWaveFile('{q(path)}');$s.Speak('{text.replace(chr(39), chr(39)*2)}');$s.Dispose();"
    try: 
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps], capture_output=True, text=True, check=True, timeout=15)
        return os.path.exists(path) and os.path.getsize(path) > 100
    except Exception as e: 
        print('[TTSエラー]', e)
        return False

def click():
    n = max(1, int(FS * CLICK_DURATION))
    t = np.arange(n) / FS
    return (CLICK_VOLUME * np.sin(2 * np.pi * CLICK_FREQUENCY * t) * np.linspace(1, 0, n)).astype(np.float32)

class Recorder:
    def __init__(self): 
        self.buf = []
        self.active = threading.Event()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.start_time = None
        self.rt = None
        self.thread = None

    def start(self):
        def cb(x, f, ti, status):
            a = x.copy()
            if self.active.is_set():
                self.buf.append(a)
                if self.rt is None and self.start_time and float(np.abs(a).max()) > THRESHOLD:
                    self.rt = time.perf_counter() - self.start_time
        def mon():
            try:
                with sd.InputStream(samplerate=FS, channels=1, callback=cb, blocksize=256, dtype='float32'):
                    self.ready.set()
                    while not self.stop.is_set(): 
                        time.sleep(.005)
            except Exception as e: 
                print('[音声入力エラー]', e)
                self.ready.set()
        self.thread = threading.Thread(target=mon, daemon=True)
        self.thread.start()
        self.ready.wait(3)

    def begin(self): 
        self.buf = []
        self.rt = None
        self.start_time = time.perf_counter()
        self.active.set()

    def end(self):
        self.active.clear()
        try: 
            return np.concatenate(self.buf, axis=0) if self.buf else None
        except Exception: 
            return None

    def close(self): 
        self.active.clear()
        self.stop.set()
        if self.thread: 
            self.thread.join(3)

def wav(a, p):
    try:
        if a is None or len(a) == 0: 
            return False
        sf.write(p, a, FS)
        return True
    except Exception as e: 
        print('[WAV保存エラー]', e)
        return False

def recognize(p): 
    return WHISPER_MODEL.transcribe(p, language='ja', task='transcribe', temperature=[0.0, 0.2, 0.4])['text'].strip()

def record(root, rec, max_duration):
    rec.begin()
    done = [False]
    timeout = [False]

    def end(e=None):
        if time.perf_counter() - rec.start_time >= .2: 
            done[0] = True
        return 'break'

    def abort_rec(e=None):
        TASK_ABORTED.set()
        done[0] = True
        return 'break'

    root.bind_all('<Return>', end)
    root.bind_all('<KP_Enter>', end)
    root.bind_all('<space>', end)
    root.bind_all('<Escape>', abort_rec)

    focus(root)
    while not done[0] and root.winfo_exists() and not TASK_ABORTED.is_set():
        root.update()
        root.update_idletasks()
        if time.perf_counter() - rec.start_time >= max_duration:
            timeout[0] = done[0] = True
        time.sleep(.01)

    return rec.end(), rec.rt, timeout[0]

def images():
    if not os.path.exists(IMG_FOLDER): 
        return []
    z = []
    for f in os.listdir(IMG_FOLDER):
        n, e = os.path.splitext(f)
        if e.lower() in ('.jpg', '.jpeg', '.png') and '_' in n:
            a, b = n.split('_', 1)
            if a.isdigit() and b.strip(): 
                z.append((int(a), f, b.strip()))
    return sorted(z)

def trial(path, num, answer, out, prefix, enable_hint, max_duration):
    target = action_verb(answer)
    r = {
        'hint_enabled': enable_hint,
        'max_duration_sec': max_duration,
        'attempt1_rt_sec': '', 'attempt1_result': '', 'attempt1_recognized': '', 'attempt1_timeout': '', 
        'cue_type': '', 'cue_text': '', 
        'attempt2_rt_sec': '', 'attempt2_result': '', 'attempt2_recognized': '', 'attempt2_timeout': '', 
        'final_result': '', 'final_recognized': '', 'audio_file': ''
    }
    r.update(trial_num=num, image_file=os.path.basename(path), answer_sentence=answer, target_action_verb=target)
    
    tmp1 = os.path.join(out, f'_temp_trial{num:02d}_attempt1.wav')
    tmp2 = os.path.join(out, f'_temp_trial{num:02d}_attempt2.wav')
    cuefile = os.path.join(out, f'_temp_trial{num:02d}_cue.wav')
    rec = None

    try:
        root = tk.Tk()
        root.title('Action Description')
        root.attributes('-fullscreen', True)
        root.attributes('-topmost', True)
        root.configure(bg='white')

        def abort_trial(e=None):
            TASK_ABORTED.set()
            try: root.destroy()
            except: pass
            return 'break'

        root.bind_all('<Escape>', abort_trial)

        try:
            im = Image.open(path)
            im.thumbnail((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS)
            x = ImageTk.PhotoImage(im)
            lab = tk.Label(root, image=x, bg='white')
            lab.image = x
            lab.pack(expand=True, fill=tk.BOTH)
        except Exception as e: 
            print('[画像エラー]', e)
            r['final_result'] = '画像エラー'
            try: root.destroy()
            except: pass
            return r

        root.update()
        focus(root)

        if TASK_ABORTED.is_set():
            try: root.destroy()
            except: pass
            return r

        rec = Recorder()
        rec.start()
        a1, rt1, to1 = record(root, rec, max_duration)

        if TASK_ABORTED.is_set():
            rec.close()
            try: root.destroy()
            except: pass
            return r

        r['attempt1_rt_sec'] = f'{rt1:.3f}' if rt1 is not None else ''
        r['attempt1_timeout'] = to1
        text1 = ''

        if to1: 
            res1 = 'タイムアウト'
        elif a1 is None: 
            res1 = '反応なし'
        else:
            wav(a1, tmp1)
            try:
                text1 = recognize(tmp1)
                res1 = '正解' if correct(text1, target, answer) else '不正解'
            except Exception: 
                res1 = '音声認識エラー'

        r['attempt1_result'] = res1
        r['attempt1_recognized'] = text1
        
        print(f"\n--- [試行 {num}] ---")
        print(f"正解文: {answer} / ターゲット動詞: {target}")
        print(f"認識テキスト: '{text1}' -> 判定: {res1}")

        safe = answer.replace('\\', '_').replace('/', '_').replace(':', '_')
        final = os.path.join(out, f'{prefix}_{num:02d}_{safe}.wav')

        if res1 == '正解' or not enable_hint:
            print("-> 終了します。")
            wav(a1, final)
            r.update(audio_file=final, final_result=res1, final_recognized=text1)
            rec.close()
            try: root.destroy()
            except: pass
            return r

        print("-> 不正解かつヒントありのためヒント処理を開始します。")
        cue = pre_verb_hint(answer)
        r['cue_type'] = 'pre_verb_bunsetsu_reading'
        r['cue_text'] = cue
        print(f"生成されたヒント: '{cue}'")
        
        made = tts(cue, cuefile)
        
        rec.begin()

        if made:
            winsound.PlaySound(cuefile, winsound.SND_ASYNC | winsound.SND_FILENAME)

        done = [False]
        to2 = [False]

        def end2(e=None):
            if time.perf_counter() - rec.start_time >= .2: 
                done[0] = True
            return 'break'

        def abort_rec2(e=None):
            TASK_ABORTED.set()
            done[0] = True
            return 'break'

        root.bind_all('<Return>', end2)
        root.bind_all('<KP_Enter>', end2)
        root.bind_all('<space>', end2)
        root.bind_all('<Escape>', abort_rec2)

        focus(root)
        while not done[0] and root.winfo_exists() and not TASK_ABORTED.is_set():
            root.update()
            root.update_idletasks()
            if time.perf_counter() - rec.start_time >= max_duration:
                to2[0] = done[0] = True
            time.sleep(.01)

        a2 = rec.end()
        rt2 = rec.rt
        r['attempt2_rt_sec'] = f'{rt2:.3f}' if rt2 is not None else ''
        r['attempt2_timeout'] = to2[0]
        text2 = ''

        if TASK_ABORTED.is_set():
            rec.close()
            try: root.destroy()
            except: pass
            return r

        if to2[0]: 
            res2 = 'タイムアウト'
        elif a2 is None: 
            res2 = '反応なし'
        else:
            wav(a2, tmp2)
            try:
                text2 = recognize(tmp2)
                res2 = '正解' if correct(text2, target, answer) else '不正解'
            except Exception: 
                res2 = '音声認識エラー'

        r['attempt2_result'] = res2
        r['attempt2_recognized'] = text2
        print(f"2回目認識テキスト: '{text2}' -> 判定: {res2}")

        if a1 is not None and a2 is not None:
            wav(np.concatenate([a1.flatten(), click(), a2.flatten()]), final)
            r['audio_file'] = final
        elif a2 is not None:
            wav(a2, final)
            r['audio_file'] = final

        r['final_result'] = 'cue後正解' if res2 == '正解' else f'cue後{res2}'
        r['final_recognized'] = text2
        rec.close()
        
        try: 
            root.destroy()
        except Exception: 
            pass

        return r

    finally:
        if rec:
            try: rec.close()
            except: pass
        for p in (tmp1, tmp2, cuefile):
            try:
                if os.path.exists(p): 
                    os.remove(p)
            except Exception: 
                pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "使用方法:\n"
            "python Action_Description.py "
            "[被験者ID] [被験者名]"
        )
        sys.exit(1)

    sid = sys.argv[1]
    sname = sys.argv[2] if len(sys.argv) >= 3 else ""

    # 1. Whisperモデル選択画面
    WHISPER_MODEL_NAME = select_whisper_model()
    if WHISPER_MODEL_NAME is None or TASK_ABORTED.is_set():
        sys.exit(0)

    # 2. ヒント有無＆制限時間選択画面
    settings_result = select_hint_settings()
    if settings_result is None or TASK_ABORTED.is_set():
        sys.exit(0)
    
    ENABLE_HINT, TASK_DURATION = settings_result

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    files = images()
    if not files:
        print(
            "[エラー] 刺激画像が見つかりません。\n"
            f"{IMG_FOLDER}"
        )
        sys.exit(1)

    print(f"Whisper {WHISPER_MODEL_NAME} モデルをロード中...")
    WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME, download_root=MODEL_DIR)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{sid}_{sname}" if sname else sid
    hint_tag = "hintON" if ENABLE_HINT else "hintOFF"
    prefix = (
        f"Action_Description_{hint_tag}_{stamp}_{sid}_{sname}"
        if sname
        else f"Action_Description_{hint_tag}_{stamp}_{sid}"
    )

    out = os.path.join(RESULTS_DIR, folder, prefix)
    os.makedirs(out, exist_ok=True)

    results = []

    for i, (n, f, a) in enumerate(files, start=1):
        if TASK_ABORTED.is_set():
            break

        if not instruction(
            "Enterキーを押してください\n\n"
            f"（試行 {i}/{len(files)}）"
        ):
            break

        if TASK_ABORTED.is_set():
            break

        trial_result = trial(
            os.path.join(IMG_FOLDER, f),
            n,
            a,
            out,
            prefix,
            ENABLE_HINT,
            TASK_DURATION
        )
        results.append(trial_result)

        if TASK_ABORTED.is_set():
            break

    fields = [
        "hint_enabled", "max_duration_sec",
        "trial_num", "image_file", "answer_sentence", "target_action_verb",
        "attempt1_rt_sec", "attempt1_result", "attempt1_recognized", "attempt1_timeout",
        "cue_type", "cue_text",
        "attempt2_rt_sec", "attempt2_result", "attempt2_recognized", "attempt2_timeout",
        "final_result", "final_recognized", "audio_file"
    ]

    csvfile = os.path.join(
        out,
        f"{prefix}_action_description_results.csv"
    )

    with open(csvfile, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("")
    if TASK_ABORTED.is_set():
        print("実験が中断されました。")
    else:
        print("実験完了")
    print(f"結果CSV: {csvfile}")
    print(f"出力フォルダー: {out}")