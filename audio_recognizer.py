import io
import wave
import pyaudio
import numpy as np
import threading
import queue
import time
from faster_whisper import WhisperModel

class AudioRecognizer:
    def __init__(self, mic_id=1, rate=16000, channels=1, chunk=1024, model_size="small", text_analyzer=None):
        self.mic_id = mic_id
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.text_analyzer = text_analyzer
        
        self.volume_score = 0.0
        self.is_running = True

        self.audio_buffer = bytearray()
        self.buffer_lock = threading.Lock()
        self.max_buffer_sec = 3.5
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.mic_id,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            print(f"[AudioRecognizer]: マイクオープンエラー: {e}")

        try:
            target_model = "turbo" if model_size == "small" else model_size
            print(f"[Whisper]: GPU(CUDA) モードでモデル '{target_model}' を読み込み中...")
            self.whisper_model = WhisperModel(target_model, device="cuda", compute_type="float16")
            print("[Whisper]: CUDA (GPU) 高速モードで起動しました。")
        except Exception:
            print(f"[Whisper]: CPU モードでモデル '{model_size}' を読み込み中...")
            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("[Whisper]: CPU モードで起動しました。")

        self.on_text_callback = None
        self.last_recognized_text = ""

        threading.Thread(target=self._audio_capture_loop, daemon=True).start()
        threading.Thread(target=self._streaming_transcribe_loop, daemon=True).start()

    def start_speech_to_text(self, on_text_detected_callback):
        """文字起こしコールバックの登録"""
        self.on_text_callback = on_text_detected_callback

    def _extract_acoustic_features(self, raw_bytes, text):
        """直近バッファ(PCM)とテキストから統合プロンプト用音響特徴（トーン・話速・強さ）を抽出"""
        try:
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            duration_sec = len(raw_bytes) / (self.rate * self.channels * 2)

            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))

            char_count = len(text.replace(" ", "")) if text else 1
            chars_per_sec = char_count / max(duration_sec, 0.5)

            if chars_per_sec > 6.0:
                speech_rate = "早口（焦り・緊張傾向）"
            elif chars_per_sec < 3.5:
                speech_rate = "ゆっくり・言い淀みあり"
            else:
                speech_rate = "普通"

            if rms > 4000:
                tone_pitch = "高め・張り詰めている"
            elif rms < 1200:
                tone_pitch = "低め・暗い"
            else:
                tone_pitch = "普通"

            peak_to_rms = peak / (rms + 1e-5)
            if peak_to_rms > 5.5 and rms > 2500:
                voice_intensity = "語尾や特定の単語に強めなトゲ・鋭さあり"
            elif rms > 4500:
                voice_intensity = "全体的に力みが強い"
            else:
                voice_intensity = "なし（フラット）"

            return tone_pitch, speech_rate, voice_intensity

        except Exception as e:
            print(f"[Acoustic Feature Extraction Error]: {e}")
            return "普通", "普通", "なし"

    def _audio_capture_loop(self):
        """マイクからの音声入力を取得し、熱量計算とリングバッファ積載を常時実行"""
        noise_threshold = 30.0
        max_volume = 3000.0
        max_bytes = int(self.rate * self.channels * 2 * self.max_buffer_sec)

        while self.is_running and self.stream:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                if not data:
                    continue

                audio_data = np.frombuffer(data, dtype=np.int16)
                if self.channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                adj_rms = max(0.0, rms - noise_threshold)
                score = min(1.0, adj_rms / (max_volume - noise_threshold))
                self.volume_score = float(round(score, 3))

                with self.buffer_lock:
                    self.audio_buffer.extend(data)
                    if len(self.audio_buffer) > max_bytes:
                        self.audio_buffer = self.audio_buffer[-max_bytes:]

            except Exception:
                pass

    def _streaming_transcribe_loop(self):
        """0.3秒サイクルで直近のバッファ音声を即時推論（字幕ストリーミング＋感情解析）"""
        min_bytes_to_process = int(self.rate * self.channels * 2 * 0.8)

        while self.is_running:
            time.sleep(0.3)

            if self.volume_score < 0.05:
                continue

            with self.buffer_lock:
                if len(self.audio_buffer) < min_bytes_to_process:
                    continue
                raw_bytes = bytes(self.audio_buffer)

            try:
                wav_io = io.BytesIO()
                with wave.open(wav_io, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(self.rate)
                    wf.writeframes(raw_bytes)
                wav_io.seek(0)

                segments, _ = self.whisper_model.transcribe(
                    wav_io,
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=200, threshold=0.4),
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    initial_prompt="リアルタイムの日本語字幕出力です。"
                )

                text = "".join([segment.text for segment in segments]).strip()

                if text and text != self.last_recognized_text:
                    self.last_recognized_text = text
                    
                    if self.on_text_callback:
                        self.on_text_callback(text)

                    if self.text_analyzer:
                        tone_pitch, speech_rate, voice_intensity = self._extract_acoustic_features(raw_bytes, text)
                        
                        self.text_analyzer.analyze_async(
                            text=text,
                            tone_pitch=tone_pitch,
                            speech_rate=speech_rate,
                            voice_intensity=voice_intensity
                        )

            except Exception as e:
                print(f"[Streaming Whisper Error]: {e}")

    def stop(self):
        self.is_running = False
        try:
            if hasattr(self, 'stream') and self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
        except Exception:
            pass

        try:
            if hasattr(self, 'p') and self.p:
                self.p.terminate()
        except Exception:
            pass