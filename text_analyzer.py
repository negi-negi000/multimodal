import datetime
import json
import os
import threading
from litellm import completion
import pandas as pd
from janome.tokenizer import Tokenizer


class TextAnalyzer:

  def __init__(
      self,
      api_key=None,
      model_name="gemini/gemini-3.5-flash-lite",
      custom_prompt="",
      csv_path="JIWC-A_2018.csv",
      mode="presentation",
  ):
    self.t = Tokenizer()
    self.api_key = api_key
    self.model_name = model_name if model_name else "gemini/gemini-3.5-flash-lite"

    if self.api_key and self.api_key.strip():
      self._set_api_key_env(self.model_name, self.api_key.strip())

    self.last_res = {
        "Joy": 0,
        "Sadness": 0,
        "Anger": 0,
        "Fear": 0,
        "Neutral": 1.0,
    }
    self.text_only_res = {
        "Joy": 0,
        "Sadness": 0,
        "Anger": 0,
        "Fear": 0,
        "Neutral": 1.0,
    }
    self.last_text = "（音声入力を待っています...）"

    self.score = 80
    self.advice = (
        "AI解析スキップ中（ローカル解析のみ）"
        if not self.api_key
        else "発話を待っています..."
    )
    self.is_processing = False
    self.has_new_log = False

    self.current_mode = mode
    self.prompts = {
        "presentation": (
            "あなたはプレゼンのプロです。聞き手に対する論理性の高さ、説得力、わかりやすさを重視して評価してください。"
        ),
        "interview": (
            "あなたは採用面接官です。面接での適切な敬語、自信、結論ファーストで話せているかを重視して評価してください。"
        ),
        "casual": (
            "あなたは話しやすい友人です。話の面白さ、親しみやすさ、共感度を重視して評価してください。"
        ),
        "custom": (
            custom_prompt
            if custom_prompt.strip()
            else (
                "あなたは優秀なアナリストです。発言の説得力と感情表現を評価してください。"
            )
        ),
    }

    self.emotion_dict = {}
    self.load_csv_dictionary(csv_path)

  def _set_api_key_env(self, model_name, api_key):
    """モデルの種類に応じて自動で対応する環境変数にAPIキーをセット"""
    if "gemini" in model_name:
      os.environ["GEMINI_API_KEY"] = api_key
    elif "gpt" in model_name or "openai" in model_name:
      os.environ["OPENAI_API_KEY"] = api_key
    elif "claude" in model_name or "anthropic" in model_name:
      os.environ["ANTHROPIC_API_KEY"] = api_key

  def set_mode(self, mode):
    if mode in self.prompts:
      self.current_mode = mode

  def set_custom_prompt(self, prompt_text):
    """カスタムプロンプトを動的に書き換える"""
    if prompt_text.strip():
      self.prompts["custom"] = prompt_text
      print("カスタムプロンプトを更新しました。")

  def load_csv_dictionary(self, csv_path):
    try:
      df = pd.read_csv(csv_path)
      df["Fear"] = df["Anxiety"]
      df["Anger_Total"] = df["Anger"] + df["Disgust"]
      self.emotion_dict = (
          df.set_index("Words")[["Joy", "Sadness", "Anger_Total", "Fear"]]
          .rename(columns={"Anger_Total": "Anger"})
          .to_dict("index")
      )
    except Exception:
      pass

  def analyze_async(
      self,
      text,
      tone_pitch="普通",
      speech_rate="普通",
      voice_intensity="なし",
  ):
    """非同期で解析処理を実行（音響特徴の入力に対応）"""
    if self.is_processing or not text.strip():
      return
    self.last_text = text
    threading.Thread(
        target=self._analyze_logic,
        args=(text, tone_pitch, speech_rate, voice_intensity),
        daemon=True,
    ).start()

  def _analyze_logic(self, text, tone_pitch, speech_rate, voice_intensity):
    self.is_processing = True
    try:
      dict_scores = {
          "Joy": 0.0,
          "Sadness": 0.0,
          "Anger": 0.0,
          "Fear": 0.0,
          "Neutral": 0.0,
      }
      tokens = self.t.tokenize(text)
      words = [token.surface for token in tokens]

      matched_count = 0
      for word in words:
        if word in self.emotion_dict:
          scores = self.emotion_dict[word]
          for emo in ["Joy", "Sadness", "Anger", "Fear"]:
            dict_scores[emo] += scores[emo]
          matched_count += 1

      if matched_count > 0:
        total_sum = sum(dict_scores.values())
        if total_sum > 0:
          for emo in ["Joy", "Sadness", "Anger", "Fear"]:
            dict_scores[emo] = round(dict_scores[emo] / total_sum, 3)
        else:
          dict_scores["Neutral"] = 1.0
      else:
        dict_scores["Neutral"] = 1.0

      self.text_only_res = dict_scores.copy()
      self.last_res = dict_scores.copy()

      if self.api_key or "ollama" in self.model_name:
        try:
          mode_instruction = self.prompts.get(
              self.current_mode, self.prompts["presentation"]
          )

          prompt_score_template = """
                                [Role]
                                あなたは音声波形と発話テキストから、話者の隠れた心理状態を鋭く見抜くプロの感情分析AIです。
                                {mode_instruction}
                                [Instruction]
                                発言内容と音響特徴（声のトーン・話速・強さ）を総合分析し、過度な平滑化（無難な数値分散）を避け、メリハリのある感情割合を算出してください。
                                [Evaluation Rules]
                                1. 【メリハリの強制】主要でない感情には安易に数値を配分せず、「0.00」または極小値（0.05以下）に設定してください。
                                2. 【強がり検知】言葉が前向き（例:「巻き返します」）でも、声に焦りや硬さがあれば Fear を最優先で高く評価してください。
                                3. 【隠れた怒り検知】語尾の強さやトゲがある場合、謝罪文であっても Anger を正しく加算してください。                                    
                                [Input]
                                発言テキスト: 「{text}」
                                音響特徴: 声のトーン={tone_pitch}, 話速={speech_rate}, 強調・トゲ={voice_intensity}
                                [Constraints]
                                1. Joy + Sadness + Anger + Fear + Neutral = 1.00 になるように設定すること。
                                2. 各感情値は float 値とし、score は 0〜100 の整数値（客観的な安心度・順調度）で評価すること。
                                [JSON Schema]
                                {{
                                "Joy": float,
                                "Sadness": float,                                    
                                "Anger": float,
                                "Fear": float,
                                "Neutral": float,
                                "score": int
                                }}"""

          prompt_score = prompt_score_template.format(
              mode_instruction=mode_instruction,
              text=text,
              tone_pitch=tone_pitch,
              speech_rate=speech_rate,
              voice_intensity=voice_intensity,
          )

          res_score = completion(
              model=self.model_name,
              messages=[{"role": "user", "content": prompt_score}],
              response_format={"type": "json_object"},
              temperature=0.1,
          )

          score_res_text = res_score.choices[0].message.content.strip()
          llm_score_data = json.loads(score_res_text)

          final_res = {}
          for emo in ["Joy", "Sadness", "Anger", "Fear", "Neutral"]:
            final_res[emo] = round(
                llm_score_data.get(emo, 0) * 0.7 + dict_scores[emo] * 0.3, 3
            )
          self.last_res = final_res

          self.score = llm_score_data.get("score", 80)

          prompt_advice_template = """
                                [Role]
                                {mode_instruction}
                                [Instruction]
                                発言者の「発言テキスト」および「分析された統合感情データ」に基づき、相手の状況・心境に寄り添った最適なアドバイスを生成してください。
                                プロンプト外の解説やMarkdown表記は含めず、指定のJSONフォーマットでのみ出力してください。
                                [Input Data]
                                発言: 「{text}」
                                状況スコア: {integrated_score} / 100 点
                                主たる感情状態: Fear={integrated_fear}, Sadness={integrated_sadness}, Neutral={integrated_neutral}
                                [Advice Guidelines]
                                  - スコアが低い（無理・焦りがある）場合：一人で抱え込ませず、具体策の整理や周囲への相談を促す優しいサポートを行う。
                                  - スコアが高い（安定している）場合：取り組みを肯定し、継続を促すコメントを行う。
                                [Constraints]
                                  1. advice は必ず 50文字以内 とすること。
                                [JSON Schema]
                                  {{
                                    "advice": string
                                  }}"""

          prompt_advice = prompt_advice_template.format(
              mode_instruction=mode_instruction,
              text=text,
              score=self.score,
              fear=final_res["Fear"],
              sadness=final_res["Sadness"],
              anger=final_res["Anger"],
              neutral=final_res["Neutral"],
          )

          res_advice = completion(
              model=self.model_name,
              messages=[{"role": "user", "content": prompt_advice}],
              response_format={"type": "json_object"},
              temperature=0.3,
          )

          advice_res_text = res_advice.choices[0].message.content.strip()
          llm_advice_data = json.loads(advice_res_text)

          self.advice = llm_advice_data.get(
              "advice", "順調なコミュニケーションです。"
          )

        except Exception as e:
          print(f"LiteLLM ({self.model_name}) 解析エラー: {e}")
          self.advice = "APIエラー（辞書解析のみ動作中）"
      else:
        self.advice = "AI解析スキップモードで動作中"

      self.has_new_log = True

    except Exception as e:
      print(f"解析全般エラー: {e}")
    finally:
      self.is_processing = False
