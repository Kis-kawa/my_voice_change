import whisper
import librosa
import numpy as np
import requests
import json
import pykakasi
import sounddevice as sd
import soundfile as sf
import subprocess

AUDIO_FILE = "your_voice.wav"
VOICEVOX_URL = "http://127.0.0.1:50021"
SPEAKER_ID = 3 # ずんだもん（ノーマル）のID

kks = pykakasi.kakasi()

def count_moras(hiragana_text):
    """ひらがなテキストから正確なモーラ数を計算する"""
    # ァィゥェォ、ャュョなどの小書き文字は、直前の文字とくっついて1モーラになるため
    # 全体の文字数から小書き文字の数を引くことで正確なモーラ数が出ます
    small_kanas = ['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ゃ', 'ゅ', 'ょ', 'ゎ']
    mora_count = len(hiragana_text)
    for kana in small_kanas:
        mora_count -= hiragana_text.count(kana)
    return mora_count


def get_pitch_at_time(target_time, times_array, f0_array):
    """指定した時間に最も近いF0（ピッチ）を取得する"""
    idx = np.abs(times_array - target_time).argmin()
    pitch = f0_array[idx]
    # 無声音などでピッチが取れない(NaN)場合は0扱い、あるいは前後の値で補間する処理を入れます
    return pitch if not np.isnan(pitch) else 0.0

extracted_pitches = []
text_for_voicevox = ""

# ==========================================
# 0. マイク録音（新規追加！）
# ==========================================
DURATION = 5  # 録音する時間（秒）を指定
FS = 16000    # Whisperの推奨サンプリングレート

print(f"🎤 {DURATION}秒間、マイクに向かって話してください...")
# マイクからの入力を記録
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, device=1)
sd.wait() # 録音終了まで待機
# WAVファイルとして保存
sf.write(AUDIO_FILE, recording, FS)
print("✅ 録音完了！ずんだもんに変換中...")

# --- 1. Whisperで文字とタイムスタンプを取得 ---
print("Whisperで文字起こし中...")
model = whisper.load_model("base")
transcribe_result = model.transcribe(AUDIO_FILE, word_timestamps=True)

print("librosaでピッチ抽出中...")
y, sr = librosa.load(AUDIO_FILE)
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)
times = librosa.times_like(f0, sr=sr)

# --- 2. モーラ分割とピッチのサンプリングロジック（アップデート版） ---
for segment in transcribe_result['segments']:
    for word_info in segment['words']:
        word_text = word_info['word']
        start_time = word_info['start']
        end_time = word_info['end']

        # pykakasiで漢字混じりのテキストをひらがなに変換
        kakasi_result = kks.convert(word_text)
        hiragana_word = "".join([item['hira'] for item in kakasi_result])

        # ひらがなから正確なモーラ数を取得
        mora_count = count_moras(hiragana_word)

        # VOICEVOXには元のテキスト（漢字混じりでもOK）を渡す
        text_for_voicevox += word_text

        if mora_count <= 0: continue

        # 1モーラあたりの時間を計算
        time_per_mora = (end_time - start_time) / mora_count

        for i in range(mora_count):
            # 各モーラの中央の時間を算出
            mora_center_time = start_time + (i * time_per_mora) + (time_per_mora / 2)
            pitch_hz = get_pitch_at_time(mora_center_time, times, f0)
            extracted_pitches.append(pitch_hz)

# 確認用出力
print("認識テキスト:", text_for_voicevox)
print("抽出されたピッチ推移(Hz)の数:", len(extracted_pitches))



# --- 3. VOICEVOX APIでAudioQueryを作成 ---
query_payload = {"text": text_for_voicevox, "speaker": SPEAKER_ID}
query_res = requests.post(f"{VOICEVOX_URL}/audio_query", params=query_payload)
audio_query = query_res.json()


# --- 4. アクセント（最大値）を反映し、イントネーションはデフォルトを維持 ---

# 自分の声の全体の平均ピッチを基準として計算（80Hz以下は除外）
valid_pitches = [p for p in extracted_pitches if p > 80.0]
user_mean_hz = sum(valid_pitches) / len(valid_pitches) if valid_pitches else 150.0

BASE_VOX_PITCH = 5.5  # ずんだもんの基準の高さ
K_FACTOR = 0.02       # テンションの反映度合い

mora_index = 0
# VOICEVOXのフレーズ（単語のまとまり）ごとに処理
for phrase in audio_query['accent_phrases']:
    phrase_user_pitches = []
    phrase_vox_pitches = []

    # 1. このフレーズ内の「自分の声のHz」と「VOICEVOXのデフォルトピッチ」を集める
    for mora in phrase['moras']:
        if mora_index < len(extracted_pitches):
            hz = extracted_pitches[mora_index]
            if hz > 80.0:
                phrase_user_pitches.append(hz)

        if mora.get('pitch') is not None:
            phrase_vox_pitches.append(mora['pitch'])

        mora_index += 1

    # 2. フレーズ内の「一番高い音（アクセント）」を計算してズレ（オフセット）を出す
    if phrase_user_pitches and phrase_vox_pitches:
        # 自分の声のフレーズ内MAXをVOICEVOXスケールに変換
        phrase_max_hz = max(phrase_user_pitches)
        target_max_pitch = BASE_VOX_PITCH + ((phrase_max_hz - user_mean_hz) * K_FACTOR)

        # VOICEVOXのデフォルトのフレーズ内MAX
        vox_max_pitch = max(phrase_vox_pitches)

        # デフォルトの形からどれくらい全体を上下させるか（オフセット）
        pitch_offset = target_max_pitch - vox_max_pitch

        # 3. VOICEVOXのイントネーション（形）を保ったまま、全体を上下にシフトさせる
        for mora in phrase['moras']:
            if mora.get('pitch') is not None:
                new_pitch = mora['pitch'] + pitch_offset

                # リミッター（4.5未満にすると掠れ声になるのを防ぐ）
                new_pitch = max(4.5, min(new_pitch, 6.5))
                mora['pitch'] = new_pitch

# 今回は個別に基準を計算したので、全体のピッチスケールは0に戻す
audio_query['pitchScale'] = 0.0

# --- 5. 音声合成と保存 ---
synth_payload = {"speaker": SPEAKER_ID}
synth_res = requests.post(
    f"{VOICEVOX_URL}/synthesis",
    params=synth_payload,
    json=audio_query
)

with open("zundamon_output.wav", "wb") as f:
    f.write(synth_res.content)

print("ずんだもんの音声ファイルが生成されました！")



def main():
    print("Hello from zundamon!")


if __name__ == "__main__":
    main()
