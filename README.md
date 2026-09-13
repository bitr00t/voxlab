# voxlab

A local-first German-speaking voice agent, built to run entirely on one
workstation GPU, with every layer swappable behind an interface.

The target stack is faster-whisper large-v3 at float16 for speech recognition,
Qwen3 on Ollama for the conversation, and Qwen3-TTS for speech synthesis. None of
it depends on a per-minute cloud service, which makes it a reasonable answer to
the two questions a mid-sized German company asks about voice agents: what does
it cost, and where does the audio go.

**Status: Phase 1.** The agent listens, thinks and speaks. Turn taking is manual
push-to-talk through the local microphone; automatic turn detection and barge-in
arrive with the browser transport in phase 2.

## Quick start

Without a GPU, to verify the wiring:

```powershell
.\scripts\setup-dev.ps1
.\.venv\Scripts\Activate.ps1
voxlab doctor
voxlab smoke
```

`voxlab smoke` runs one full turn using the placeholder providers. It needs no
GPU, no model weights and no microphone - it proves the wiring, not the speech.

To actually talk to it:

```powershell
.\scripts\setup-dev.ps1 -Extras dev,stt,audio
.\scripts\pull-models.ps1
Copy-Item .env.example .env   # then uncomment the phase 1 block
voxlab devices                # note the index of your microphone
voxlab talk
```

Enter starts recording, Enter stops it, `q` ends the conversation. The first run
downloads the Whisper weights, so give it a minute.

## Commands

| Command | What it does |
| --- | --- |
| `voxlab doctor` | Checks Python, GPU, CUDA libraries, Ollama and audio devices |
| `voxlab providers` | Lists every registered provider |
| `voxlab config` | Prints the effective configuration |
| `voxlab smoke` | Runs one turn and reports per-stage latency |
| `voxlab talk` | Push-to-talk conversation through microphone and speakers |
| `voxlab devices` | Lists audio devices with the index to configure |
| `voxlab tts-probe` | Reports which speech synthesis packages are installed |

## Configuration

Everything is configured through environment variables or a `.env` file, using a
double underscore for nesting. Copy `.env.example` to `.env` to start.

Provider selection is a string, never an import. Switching from the placeholder
synthesiser to the real one is one line:

```
VOXLAB__TTS__PROVIDER=qwen3
```

## Architecture

Four seams, each behind an abstract interface: an audio transport, speech
recognition, a language model, and speech synthesis.

```
AudioTransport  ->  SpeechToText  ->  LanguageModel  ->  TextToSpeech  ->  AudioTransport
 (mic / browser)   (faster-whisper)   (Qwen3/Ollama)      (Qwen3-TTS)        (playback)
```

The transport seam is the one that matters most on Windows, and the reasoning
behind it is written down in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Roadmap

| Phase | Content | Status |
| --- | --- | --- |
| 0 | Repository, provider interfaces, configuration, environment checks | done |
| 1 | Push-to-talk loop on the host: faster-whisper, Ollama, Qwen3-TTS | done |
| 2 | Real time in the browser: WebRTC transport, voice activity detection, barge-in | next |
| 3 | Retrieval-augmented answers over a document base, plus tool calling | planned |
| 4 | Evaluation harness and a provider comparison against a hosted service | planned |

## Hardware

Developed against an NVIDIA RTX A4500 (20 GB). Indicative VRAM budget for the
Phase 1 stack:

| Component | Configuration | VRAM |
| --- | --- | --- |
| faster-whisper | large-v3, float16 | ~4.5 GB |
| Qwen3 via Ollama | 8B, Q5 quantisation | ~6-7 GB |
| Qwen3-TTS | float16 | ~4 GB |

That leaves headroom on a 20 GB card. Swapping in a larger language model for
better German means giving some of it back - most cheaply by moving Whisper to
`medium` at int8.

## Speech synthesis

Local speech synthesis is the youngest part of the stack and its Python
interfaces move. Everything uncertain is confined to one module,
`voxlab.providers.tts.engine`, which obtains an engine in three ways: an
explicit factory named in the configuration, a transformers-based engine for
weights published as a Hugging Face model, or a package-level entry point tried
against a few plausible call shapes.

If `voxlab talk` cannot construct an engine, run `voxlab tts-probe` to see what
is installed, then write your own factory and point the configuration at it:

```python
# my_adapters/qwen3.py
from voxlab.providers.tts.engine import Engine, Samples

class MyEngine(Engine):
    def __init__(self, model: str, device: str) -> None:
        ...  # whatever the upstream package actually wants

    def synthesize(self, text: str, voice: str) -> Samples:
        return Samples(data=..., sample_rate=24000)

def create(model: str, device: str) -> MyEngine:
    return MyEngine(model, device)
```

```
VOXLAB__TTS__ADAPTER=my_adapters.qwen3:create
```

Fifteen lines of your own beat fighting a wrong assumption in this repository.

## Sample rates

Whisper works at 16 kHz, capture devices often do not offer that rate, and
synthesis usually emits 24 kHz. Conversion happens in `voxlab.audio.resample`
and nowhere else. The transport negotiates the capture rate with the device and
resamples when it has to; playback opens the output device at the rate of the
audio it is handed, so changing synthesiser cannot silently detune it.

`soxr` does the work and ships with the `audio` extra. Without it the code falls
back to linear interpolation, which is audible on the way out and measurable as
lost recognition accuracy on the way in - `voxlab doctor` says so rather than
leaving it to be discovered.

## Windows notes

Three things cost an evening if nobody wrote them down:

1. **faster-whisper needs cuBLAS and cuDNN.** The `stt` extra pulls the
   `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels, which avoids installing
   the CUDA Toolkit by hand. On Windows the DLLs then sit in `site-packages`
   where the loader does not look, so `voxlab.runtime.cuda_windows` registers
   those directories before CTranslate2 is imported.
2. **The cuDNN major version has to match the CTranslate2 build.** A mismatch
   fails as a missing `cudnn_ops64_9.dll`, or as a process that exits with no
   traceback at all. `voxlab doctor` reports this as a `ctranslate2` failure
   rather than letting it surface mid-call.
3. **WSL2 has no microphone.** It passes the GPU through, but there is no direct
   audio device access, so putting the models in WSL2 and the microphone on the
   host is a dead end. Phase 2 resolves this by moving audio into the browser.

## Licence

MIT. Note that model weights carry their own licences, which are not covered
here and differ between the models named above.
