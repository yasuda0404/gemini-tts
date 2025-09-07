#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini TTS バッチ生成スクリプト
--------------------------------
・.env から GEMINI_API_KEY を読み込み、Gemini TTS モデルのクライアントを初期化します
・「#機能」で与えられた JSON データ（id / voice_name / lang / instruction / text）に従って、
  各テキストを指定ボイスで読み上げ音声に変換し、output ディレクトリに保存します
・モデル: def synthesize_items
    "gemini-2.5-flash-preview-tts"（単一話者TTS対応） or
    "gemini-2.5-pro-preview-tts"（単一話者TTS対応）
・音声出力は 24kHz, 16bit, mono の PCM を WAV に書き込みます
  （公式サンプルの保存方法に準拠：wave モジュールでWAVヘッダを付与）※参照元: Speech generation ガイド

【使い方】
1) `cp .env.sample .env` で .env を作り、GEMINI_API_KEY を設定
2) `pip install -r requirements.txt`
3) `python gemini-tts.py`
   - サンプルの入力JSONはソース最下部の INPUT_JSON に埋め込み済みです
   - 外部JSONを使いたい場合は、`python gemini-tts.py input.json` のようにファイルパスを渡してください

【備考】
・TTSは入力テキストの言語を自動検出します（ja-JP 対応）。`lang`はプロンプト形成のヒントとして扱います。
"""

import os
import sys
import json
import wave
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Gemini GenAI SDK（公式）
# 参考: Speech generation (text-to-speech) / Gemini API libraries
# https://ai.google.dev/gemini-api/docs/speech-generation
# https://ai.google.dev/gemini-api/docs/libraries
from google import genai
from google.genai import types


# ===== ユーティリティ =====


def load_api_key_from_env() -> str:
    """.env を読み込み、GEMINI_API_KEY を環境変数から取得する。未設定ならエラー終了。"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.stderr.write(
            "エラー: GEMINI_API_KEY が見つかりません。.env に GEMINI_API_KEY=... を設定してください。\n"
        )
        sys.exit(1)
    # google-genai のクライアントは環境変数を自動取得するため、戻り値は確認用
    return api_key


def ensure_output_dir(path: Path) -> None:
    """出力ディレクトリが無ければ作成する。"""
    path.mkdir(parents=True, exist_ok=True)


def save_wave(
    filename: Path,
    pcm_bytes: bytes,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
) -> None:
    """生PCMバイト列を WAV 形式で保存する。"""
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)  # 2 bytes = 16bit
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)


def get_common_json(raw: str) -> Dict[str, Any]:
    """
    与えられた JSON テキストをパースし、アイテム配列（List[Dict]）を返す。
    ・ユーザ提示のJSONには全角引用符（“ ”）やトップレベル配列/辞書の揺れがあるため、正規化して受け付ける
    ・想定キー: "common" - "instruction"
    """
    s = raw.replace("“", '"').replace("”", '"')
    obj = json.loads(s)

    key = "common"
    if isinstance(obj, dict):
        if isinstance(obj.get(key), dict):
            return obj.get(key)
        return obj
    raise ValueError(
        "入力JSONの形式が正しくありません（'common'が見つかりませんでした）。"
    )


def normalize_and_parse_json(raw: str) -> List[Dict[str, Any]]:
    """
    与えられた JSON テキストをパースし、アイテム配列（List[Dict]）を返す。
    ・ユーザ提示のJSONには全角引用符（“ ”）やトップレベル配列/辞書の揺れがあるため、正規化して受け付ける
    ・想定キー: "items" / "data" / "texts" / "tts" のいずれか、またはトップレベル配列
    """
    s = raw.replace("“", '"').replace("”", '"')
    obj = json.loads(s)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ("items", "data", "texts", "tts"):
            if isinstance(obj.get(key), list):
                return obj[key]
        # 最初に見つかった配列値を返す（フォールバック）
        for v in obj.values():
            if isinstance(v, list):
                return v

    raise ValueError("入力JSONの形式が正しくありません（配列が見つかりませんでした）。")


def build_tts_prompt(item: Dict[str, Any]) -> str:
    """
    instruction と lang, text を使って TTS 用の自然言語プロンプトを作る。> 現在は, lang,textは使っていない
    ・公式ガイドは自然言語指示でトーン/ペース等を制御する方式を推奨
    """
    instruction = item.get("instruction", "Read clearly")
    # lang = item.get("lang", "").strip()
    text = item.get("text", "")

    # 明示的に言語ヒントを与える（自動検出に加え、プロンプトでも補強）
    # return f'「{instruction}」のトーンで、{lang} のテキストを読み上げてください。\n{text}'
    if text == "":
        return ""
    else:
        return f"{instruction}: {text}"


# ===== メイン処理 =====


def synthesize_items(
    common: Dict[str, Any],
    items: List[Dict[str, Any]],
    model_name: str = "gemini-2.5-flash-preview-tts",
) -> None:
    """
    各アイテムについて TTS を実行し、`output/<id>_<voice>.wav` に保存する。
    公式の単一話者TTSサンプルに準拠し、response_modalities=["AUDIO"] で音声を取得します。
    """
    out_dir = Path("output")
    ensure_output_dir(out_dir)

    # 共通項目
    temperature = 1.0
    try:
        temperature = float(common.get("temperature", 1.0))
        # print(f"temperature = {temperature}")
    except:
        sys.stderr.write(f"[Warning] common-temperature not found: {e}\n")

    # 共通インストラクションを加える
    common_prompt = ""
    try:
        common_prompt = common.get("instruction", "")
        # print(f"PROMPT: {common_prompt}")
    except Exception as e:
        # 失敗しても他アイテムの処理は続ける
        sys.stderr.write(f"[Warning] common-instruction not found: {e}\n")

    # 環境変数 GEMINI_API_KEY を自動検出して初期化（Quickstart準拠）
    client = genai.Client()

    for item in items:
        id_ = str(item.get("id", "no-id"))
        voice_name = str(item.get("voice_name", "Kore"))
        prompt = build_tts_prompt(item)
        
        if prompt == "":
            print(f"{id_} はスキップします。")
            continue    #以下の処理は行わずに次のitemへ
        
        prompt = common_prompt + prompt  # common_promptを先頭に加える
        print(f"[temp] {temperature}, [prompt] {prompt}")

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_modalities=["AUDIO"],  # 音声出力を明示
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        )
                    ),
                ),
            )

            # 公式サンプルと同様に inline_data.data からPCMを取得
            # Python SDKでは bytes が返るため、そのままWAV化できる
            pcm_bytes = response.candidates[0].content.parts[0].inline_data.data

            # filename = out_dir / f"{id_}_{voice_name}.wav"
            filename = out_dir / f"{id_}.wav"
            save_wave(filename, pcm_bytes)
            print(f"[OK] {filename} を保存しました")

        except Exception as e:
            # 失敗しても他アイテムの処理は続ける
            sys.stderr.write(
                f"[ERROR] id={id_}, voice={voice_name} の生成に失敗: {e}\n"
            )


def main():
    # 1) APIキー読み込み（環境変数に無い場合はエラー）
    load_api_key_from_env()

    # 2) 入力JSONの取得
    #    引数があればファイルから読み込み、無ければサンプルJSONを使用
    if len(sys.argv) >= 2:
        json_path = Path(sys.argv[1])
        raw = json_path.read_text(encoding="utf-8")
    else:
        # —— サンプル入力（ユーザ提示JSONを正しい形式に整形）——
        raw = r"""{
          "common":{
            "instruction": "Read loud."
          },
          "items": [
            {
              "id": "01-01",
              "voice_name": "Kore",
              "lang": "ja-JP",
              "instruction": "Read clearly",
              "text": "このテキストを読み上げて下さい"
            },
            {
              "id": "01-02",
              "voice_name": "Puck",
              "lang": "ja-JP",
              "instruction": "Read brightly",
              "text": "こちらの文章をよみあげてください！"
            }
          ]
        }"""

    try:
        common = get_common_json(raw)
        print(common.get("instruction"))
    except Exception as e:
        sys.stderr.write(f"入力JSONの解析に失敗しました: {e}\n")
        sys.exit(1)

    try:
        items = normalize_and_parse_json(raw)
    except Exception as e:
        sys.stderr.write(f"入力JSONの解析に失敗しました: {e}\n")
        sys.exit(1)

    # 3) 合成の実行
    # gemini-2.5-pro-preview-tts より、flash-preview-tts の方が安定している
    synthesize_items(common, items, model_name="gemini-2.5-flash-preview-tts")


if __name__ == "__main__":
    main()
