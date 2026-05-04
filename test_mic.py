import sounddevice as sd
import numpy as np

FS = 16000
DURATION = 3

# 現在認識されているオーディオデバイス一覧を表示
print("=== 🎧 認識されているデバイス一覧 ===")
print(sd.query_devices())
print("===================================")

print("\n🎤 3秒間テスト録音します。何か喋ってください...")
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
sd.wait()

# 録音されたデータの「音量の最大値」を計算
max_volume = np.max(np.abs(recording))
print(f"\n📊 録音された最大音量: {max_volume:.5f}")

if max_volume == 0.0:
    print("❌ 完全に無音です。マイクの許可設定か、デバイス指定が間違っています。")
elif max_volume < 0.01:
    print("⚠️ 音が小さすぎます。Mac自体のマイク入力音量を確認してください。")
else:
    print("✅ 音声は正常に拾えています！本番の main.py でも動くはずです。")
