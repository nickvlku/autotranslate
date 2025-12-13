"""Subtitle translation using LLMs (DeepSeek and Ollama)."""

import json
import re
import time

import ollama
from openai import OpenAI

from .config import Config
from .models import Subtitle


TRANSLATION_SYSTEM_PROMPT = """You are a professional subtitle translator. Translate the following subtitles from {source_lang} to {target_lang}.

Rules:
1. Preserve the exact JSON structure - only translate the "text" field
2. Keep translations natural and appropriate for subtitles (concise, readable)
3. Preserve any speaker indicators like [Speaker 1:] if present
4. Do not add or remove subtitle entries
5. Return ONLY valid JSON, no other text

Input format: [{{"index": 1, "text": "original text"}}, ...]
Output format: [{{"index": 1, "text": "translated text"}}, ...]"""

MAX_RETRIES = 3
RETRY_DELAY = 1.0


def _batch_subtitles(subtitles: list[Subtitle], max_chars: int = 3000) -> list[list[Subtitle]]:
    """Split subtitles into batches to stay within token limits."""
    batches = []
    current_batch = []
    current_chars = 0

    for sub in subtitles:
        sub_chars = len(sub.text) + 50
        if current_chars + sub_chars > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(sub)
        current_chars += sub_chars

    if current_batch:
        batches.append(current_batch)

    return batches


def _fix_json(text: str) -> str:
    """Attempt to fix common JSON issues from LLM output."""
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)

    # Find the JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        raise ValueError("Could not find JSON array in response")

    json_str = match.group()

    # Fix common issues
    # 1. Trailing commas before ] or }
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)

    # 2. Single quotes to double quotes (but not inside strings)
    # This is tricky, so only do it if double quotes aren't present
    if '"index"' not in json_str and "'index'" in json_str:
        json_str = json_str.replace("'", '"')

    # 3. Unescaped newlines in strings - replace with space
    # Match content between "text": " and the closing "
    def fix_newlines(m):
        content = m.group(1)
        content = content.replace('\n', ' ').replace('\r', '')
        return f'"text": "{content}"'

    json_str = re.sub(r'"text":\s*"([^"]*(?:\\.[^"]*)*)"', fix_newlines, json_str)

    return json_str


def _parse_translation_response(response: str, original_subs: list[Subtitle]) -> list[Subtitle]:
    """Parse LLM translation response and merge with original subtitles."""
    try:
        json_str = _fix_json(response)
        translated = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid JSON in response: {e}\nResponse: {response[:500]}")

    # Create lookup by index
    trans_by_index = {t["index"]: t["text"] for t in translated}

    result = []
    for sub in original_subs:
        translated_text = trans_by_index.get(sub.index, sub.text)
        result.append(
            Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                text=translated_text,
            )
        )

    return result


def _translate_batch_deepseek(
    client: OpenAI,
    batch: list[Subtitle],
    source_lang: str,
    target_lang: str,
) -> list[Subtitle]:
    """Translate a single batch with retry logic."""
    input_data = [{"index": s.index, "text": s.text} for s in batch]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": TRANSLATION_SYSTEM_PROMPT.format(
                            source_lang=source_lang, target_lang=target_lang
                        ),
                    },
                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                ],
                temperature=0.3,
            )

            result_text = response.choices[0].message.content
            return _parse_translation_response(result_text, batch)

        except ValueError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise

    raise last_error


def translate_deepseek(
    subtitles: list[Subtitle],
    config: Config,
    source_lang: str,
    target_lang: str,
    on_progress: callable = None,
) -> list[Subtitle]:
    """Translate subtitles using DeepSeek API."""
    if not config.deepseek_api_key:
        raise RuntimeError("DeepSeek API key not configured")

    client = OpenAI(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
    )

    batches = _batch_subtitles(subtitles)
    translated = []

    for i, batch in enumerate(batches):
        if on_progress:
            on_progress(i + 1, len(batches))

        batch_translated = _translate_batch_deepseek(
            client, batch, source_lang, target_lang
        )
        translated.extend(batch_translated)

    return translated


def _translate_batch_openai(
    client: OpenAI,
    batch: list[Subtitle],
    source_lang: str,
    target_lang: str,
    model: str = "gpt-5.2-chat-latest",
) -> list[Subtitle]:
    """Translate a single batch using OpenAI with retry logic."""
    input_data = [{"index": s.index, "text": s.text} for s in batch]

    # GPT-5.2 models don't support custom temperature
    is_gpt5 = model.startswith("gpt-5")

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": TRANSLATION_SYSTEM_PROMPT.format(
                            source_lang=source_lang, target_lang=target_lang
                        ),
                    },
                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                ],
            }
            if not is_gpt5:
                kwargs["temperature"] = 0.3

            response = client.chat.completions.create(**kwargs)

            result_text = response.choices[0].message.content
            return _parse_translation_response(result_text, batch)

        except ValueError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise

    raise last_error


def translate_openai(
    subtitles: list[Subtitle],
    config: Config,
    source_lang: str,
    target_lang: str,
    model: str = "gpt-5.2-chat-latest",
    on_progress: callable = None,
) -> list[Subtitle]:
    """Translate subtitles using OpenAI API."""
    if not config.openai_api_key:
        raise RuntimeError("OpenAI API key not configured")

    client = OpenAI(api_key=config.openai_api_key)

    batches = _batch_subtitles(subtitles)
    translated = []

    for i, batch in enumerate(batches):
        if on_progress:
            on_progress(i + 1, len(batches))

        batch_translated = _translate_batch_openai(
            client, batch, source_lang, target_lang, model
        )
        translated.extend(batch_translated)

    return translated


def _translate_batch_ollama(
    batch: list[Subtitle],
    source_lang: str,
    target_lang: str,
    model: str,
) -> list[Subtitle]:
    """Translate a single batch with retry logic."""
    input_data = [{"index": s.index, "text": s.text} for s in batch]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": TRANSLATION_SYSTEM_PROMPT.format(
                            source_lang=source_lang, target_lang=target_lang
                        ),
                    },
                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                ],
                options={"temperature": 0.3},
            )

            result_text = response["message"]["content"]
            return _parse_translation_response(result_text, batch)

        except ValueError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise

    raise last_error


def translate_ollama(
    subtitles: list[Subtitle],
    source_lang: str,
    target_lang: str,
    model: str = "llama3.1:8b",
    on_progress: callable = None,
) -> list[Subtitle]:
    """Translate subtitles using local Ollama."""
    batches = _batch_subtitles(subtitles, max_chars=2000)
    translated = []

    for i, batch in enumerate(batches):
        if on_progress:
            on_progress(i + 1, len(batches))

        batch_translated = _translate_batch_ollama(
            batch, source_lang, target_lang, model
        )
        translated.extend(batch_translated)

    return translated


LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
}


def get_language_name(code: str) -> str:
    """Get full language name from code."""
    return LANGUAGE_NAMES.get(code.lower(), code)


def translate(
    subtitles: list[Subtitle],
    config: Config,
    source_lang: str,
    target_lang: str,
    provider: str = "deepseek",
    ollama_model: str = "llama3.1:8b",
    openai_model: str = "gpt-5.2-chat-latest",
    on_progress: callable = None,
) -> list[Subtitle]:
    """Translate subtitles using specified provider."""
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)

    if provider == "deepseek":
        return translate_deepseek(subtitles, config, source_name, target_name, on_progress)
    elif provider == "openai":
        return translate_openai(subtitles, config, source_name, target_name, openai_model, on_progress)
    elif provider == "ollama":
        return translate_ollama(subtitles, source_name, target_name, ollama_model, on_progress)
    else:
        raise ValueError(f"Unknown translation provider: {provider}")
