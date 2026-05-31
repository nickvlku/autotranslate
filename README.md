# autotranslate

A CLI tool that translates video/audio to subtitles in another language. It extracts audio, transcribes it using Whisper, and translates the result using an LLM.

## Features

- **Audio extraction** from video files using ffmpeg
- **Transcription** via OpenAI Whisper, Groq, OpenRouter, faster-whisper (local), whisper-cli (local whisper.cpp), or Qwen3-ASR (local, with forced-aligner timestamps)
- **Translation** using DeepSeek, OpenAI, OpenRouter, or Ollama (local)
- **SRT output** with proper timestamps
- **Embed subtitles** back into the video as soft subtitle tracks (`--embed`)
- Automatic chunking for large audio files
- Audio volume boost for quiet content
- Resume from intermediate files if processing fails

## Requirements

- Python 3.12+
- ffmpeg installed and in PATH
- API keys for cloud services (optional if using local models)
- `whisper-cli` in PATH plus a ggml model file (only if using `--whisper whisper-cli`)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autotranslate.git
cd autotranslate

# Install with uv
uv sync

# Or install with pip
pip install -e .

# Optional: local Qwen3-ASR transcription (--whisper qwen-asr). Heavy (pulls in
# torch + ~4.6 GB of models on first run). Runs on Apple Silicon (MPS) or CPU.
uv sync --extra qwen      # or: pip install -e ".[qwen]"
```

## Configuration

Set environment variables for the services you want to use:

```bash
# For DeepSeek translation (default)
export DEEPSEEK_API_KEY="your-deepseek-key"

# For OpenAI Whisper transcription and/or translation
export OPENAI_API_KEY="your-openai-key"
# Optional: target an OpenAI-compatible endpoint such as LM Studio
# (the --openai-base-url flag overrides this)
export OPENAI_BASE_URL="http://localhost:1234/v1"

# For OpenRouter (Claude, GPT-4, etc.)
export OPENROUTER_API_KEY="your-openrouter-key"

# For Groq (fast Whisper transcription)
export GROQ_API_KEY="your-groq-key"
```

See `.env.example` for a template.

## Usage

### Basic usage

```bash
# Translate Japanese video to English subtitles
autotranslate video.mp4 --from ja --to en

# Translate Chinese audio to English
autotranslate audio.wav --audio --from zh --to en

# Translate existing transcript
autotranslate transcript.ja.srt --transcript --from ja --to en
```

### Options

```
autotranslate INPUT_PATH --from SOURCE_LANG --to TARGET_LANG [OPTIONS]

Arguments:
  INPUT_PATH              Video, audio (--audio), or SRT file (--transcript)

Required Options:
  --from TEXT             Source language code (e.g., ja, zh, es)
  --to TEXT               Target language code (e.g., en, fr, de)

Input Options:
  --audio, -a             Treat input as audio file (skip extraction)
  --transcript, -t        Treat input as SRT (skip extraction and transcription)

Provider Options:
  --whisper [openai|groq|openrouter|local|whisper-cli|qwen-asr]
                          Transcription provider (default: openai). qwen-asr is
                          local Qwen3-ASR + forced aligner (needs the [qwen] extra)
  --whisper-model TEXT    Whisper model: an API model name (e.g. whisper-1,
                          whisper-large-v3-turbo, large-v3), or for
                          --whisper whisper-cli a ggml model file path
                          (e.g. ./ggml-large-v3.bin); required for whisper-cli
  --vad-model TEXT        whisper-cli only: enable Voice Activity Detection
                          with this VAD model file (e.g. ./ggml-silero-v6.2.0.bin)
  --llm [deepseek|openai|openrouter|ollama]
                          LLM provider for translation (default: deepseek)
  --ollama-model TEXT     Ollama model to use (default: llama3.1:8b)
  --openai-model TEXT     OpenAI model to use (default: gpt-4o)
  --openai-base-url TEXT  Override the OpenAI-compatible base URL (e.g.
                          http://localhost:1234/v1 for LM Studio); applies to
                          the openai provider for translation and transcription
  --openai-extra-body TEXT
                          Extra JSON merged into openai translation requests
                          (OpenAI-compatible servers), e.g. to disable thinking
                          on models that support a toggle:
                          '{"chat_template_kwargs": {"enable_thinking": false}}'
  --openrouter-model TEXT OpenRouter model (default: anthropic/claude-3.5-sonnet)

Output Options:
  --output, -o PATH       Output SRT file path
  --cleanup               Delete intermediate files on success
  --embed                 Embed the subtitle track(s) into the source video as
                          a new <name>.subbed.<ext> file (soft subs, no
                          re-encode). Two tracks are added (source transcript +
                          target translation), each titled "Language (model)".
                          Video input only.

Audio Options:
  --boost FLOAT           Boost audio volume in dB (e.g., 10 for +10dB)
```

### Examples

```bash
# Use local Whisper and Ollama (fully offline)
autotranslate video.mp4 --from ja --to en --whisper local --llm ollama

# Use local whisper-cli (whisper.cpp) with a ggml model file
autotranslate video.mp4 --from ja --to en --whisper whisper-cli --whisper-model ./ggml-large-v3.bin

# whisper-cli with Voice Activity Detection (filters non-speech)
autotranslate video.mp4 --from ja --to en --whisper whisper-cli \
  --whisper-model ./ggml-large-v3.bin --vad-model ./ggml-silero-v6.2.0.bin

# Local Qwen3-ASR (requires the [qwen] extra). Word-level timestamps from the
# forced aligner are grouped into cues at sentence boundaries. Runs on MPS/CPU.
autotranslate video.mp4 --from ja --to en --whisper qwen-asr

# Use Groq for fast transcription and OpenRouter for translation
autotranslate video.mp4 --from ja --to en --whisper groq --llm openrouter

# Use OpenAI for translation with specific model
autotranslate video.mp4 --from ja --to en --llm openai --openai-model gpt-4o

# Use specific Whisper model
autotranslate video.mp4 --from ja --to en --whisper groq --whisper-model whisper-large-v3-turbo

# Translate with a local OpenAI-compatible server (e.g. LM Studio)
autotranslate video.mp4 --from ja --to en \
  --llm openai --openai-model my-local-model --openai-base-url http://localhost:1234/v1

# "Thinking" models work out of the box (reasoning is stripped before parsing).
# For models that support a thinking toggle (e.g. Qwen), you can also disable it:
autotranslate video.mp4 --from ja --to en \
  --llm openai --openai-model my-thinking-model --openai-base-url http://localhost:1234/v1 \
  --openai-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'

# Boost quiet audio before transcription
autotranslate video.mp4 --from ja --to en --boost 10

# Clean up intermediate files after successful translation
autotranslate video.mp4 --from ja --to en --cleanup

# Embed the subtitles back into the video (creates video.subbed.mp4)
autotranslate video.mp4 --from ja --to en --embed

# Bake off several models: one translation (SRT + subtitle track) per model.
# Progressive - if one model fails, the others still complete and get embedded.
autotranslate video.mp4 --from ja --to en --embed \
  --llm openai --openai-base-url http://localhost:1234/v1 \
  --openai-model "model-a,model-b,model-c"
```

### Supported Languages

Common language codes: `en` (English), `ja` (Japanese), `zh` (Chinese), `ko` (Korean), `es` (Spanish), `fr` (French), `de` (German), `it` (Italian), `pt` (Portuguese), `ru` (Russian), `ar` (Arabic), `hi` (Hindi), `th` (Thai), `vi` (Vietnamese)

## Output Files

The tool generates intermediate files with model names that can be used to resume processing:

- `{name}.{whisper-model}.{source-lang}.srt` - Transcription output
- `{name}.{llm-model}.{target-lang}.srt` - Final translated subtitles

For example:
- `video.whisper-1.ja.srt` - OpenAI Whisper transcription
- `video.deepseek.en.srt` - DeepSeek translation
- `video.gpt-4o.en.srt` - GPT-4o translation

With `--embed`, a `{name}.subbed.{ext}` video is also written, containing one
soft subtitle track per successful model (plus the source transcript), each
titled `Language (model)`. Video and audio are stream-copied (no re-encode); the
track title is stored as the MP4 track `name` (or MKV `title`) that players
display. During a multi-model bake-off the file is re-muxed and atomically
replaced after each model, so an interrupted run never loses finished tracks.

If processing fails, the tool shows a command to retry from the last successful step.

## Local Mode

For fully offline operation:

1. Install Ollama and pull a model: `ollama pull llama3.1:8b`
2. Use local Whisper (downloads model on first run): `--whisper local`
3. Use Ollama for translation: `--llm ollama`

```bash
autotranslate video.mp4 --from ja --to en --whisper local --llm ollama
```

Alternatively, transcribe with `whisper-cli` (whisper.cpp). Install it so the
`whisper-cli` binary is on your PATH, download a ggml model (e.g.
`ggml-large-v3.bin`), and pass its path via `--whisper-model`:

```bash
autotranslate video.mp4 --from ja --to en \
  --whisper whisper-cli --whisper-model ./ggml-large-v3.bin --llm ollama
```

## License

GNU GENERAL PUBLIC LICENSE V3
