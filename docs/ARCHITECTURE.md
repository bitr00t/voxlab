# Architecture

## The four seams

Everything that could plausibly be replaced sits behind an interface:

| Seam | Interface | Phase 0 placeholder | Target |
| --- | --- | --- | --- |
| Audio in and out | `AudioTransport` | `null` | `local` (phase 1), browser (phase 2) |
| Speech recognition | `SpeechToText` | `echo` | faster-whisper large-v3, float16 |
| Conversation | `LanguageModel` | `static` | Qwen3 on Ollama |
| Speech synthesis | `TextToSpeech` | `silent` | Qwen3-TTS |

Providers register themselves under a string name; the pipeline never imports a
concrete class. Two consequences that were the point of doing it this way:

* Heavy dependencies stay optional. A machine without CUDA can still import the
  package, run the tests and execute a turn.
* Swapping a provider is a configuration change, which makes an A/B comparison
  between a local model and a hosted service a measurement rather than a
  rewrite. That comparison is the deliverable of Phase 4.

## Why audio moves into the browser in Phase 2

The obvious way to run local models on Windows is to put them in WSL2 or Docker,
where the tooling is friendlier. GPU passthrough into WSL2 works. Microphone
access does not - there is no direct path from a WSL2 process to a host audio
device. So the tempting setup, models in WSL2 and microphone on the host, is a
dead end that only reveals itself after the models are already running.

There are three ways out, and only one of them is any good:

1. Keep everything native on Windows. Works, and it is what Phase 1 does, but it
   ties the models to one machine and one Python environment.
2. Bridge audio into WSL2 over a socket or a virtual device. Fragile, and the
   added buffering eats part of the latency budget the project exists to defend.
3. Put audio in the browser. `getUserMedia` and WebRTC handle capture and
   playback, the browser's echo cancellation comes along for free, and the models
   become HTTP services that no longer care which operating system or container
   they run in.

Option 3 also produces something demonstrable: a URL that can be opened during a
screen share, rather than a recorded video.

So Phase 1 uses a `sounddevice` transport for the shortest path to a working
loop, and Phase 2 adds a WebRTC transport alongside it. Both implement the same
interface, and neither one is visible to the pipeline.

## Where voice activity detection lives

In the transport, not in the speech recognition provider. Deciding when an
utterance has ended depends on how audio arrives - buffer sizes, jitter, whether
the far side can be interrupted - and not on which model transcribes it. Putting
it in the transport keeps `SpeechToText` to a single responsibility and lets the
browser transport use a different turn detector from the local one without
touching Whisper.

## Half duplex first

Phase 0 and Phase 1 run a half-duplex loop: the agent finishes speaking before it
listens again. Barge-in makes this a full-duplex problem - playback has to be
cancellable, and incoming audio has to be monitored while output is running.
That belongs with the transport that can actually do it, so it waits for Phase 2.

## Latency accounting

`TurnMetrics` is populated from Phase 0 onwards, even though the placeholder
numbers are meaningless. The point is that later phases inherit a pipeline that
already measures itself, instead of one that gets instrumented after the latency
has already become disappointing.

The number that matters is time to first audio: speech recognition, plus the
first token from the language model, plus the first chunk from the synthesiser.
Under roughly 800 ms a conversation feels responsive. Past 1.5 seconds, callers
start talking over the agent or hang up.

One consequence is already visible in the code: `VoicePipeline._collect_reply`
drains the entire model response before synthesis starts, which is the
straightforward implementation and the wrong one. Phase 2 replaces it with
sentence-wise hand-off to the synthesiser. The measurement is in place first, so
the improvement shows up as a number.

## Why push-to-talk in phase 1

Turn taking is manual: Enter to start speaking, Enter to stop. That is a
deliberate simplification rather than a missing feature.

Deciding automatically when a caller has finished speaking is a quality problem
of its own - too eager and the agent interrupts, too patient and every exchange
gains a second of dead air. Solving it at the same time as getting three models
to produce German would have made it impossible to tell which of the two was
responsible for a bad conversation. Manual turn taking removes that variable
until the rest is known to work.

Automatic detection arrives with the browser transport in phase 2, where it
belongs anyway: it is a property of how audio arrives, not of which model
transcribes it.

## Where sample rate conversion happens

In `voxlab.audio.resample`, and nowhere else.

Three components disagree about rates: the recogniser wants 16 kHz, capture
devices frequently refuse it, and the synthesiser emits whatever it was trained
on. Two rules keep that from spreading:

* The transport negotiates with the device. It asks for 16 kHz, and when
  PortAudio refuses, falls back to the device's own rate and resamples. The
  provider above never learns that this happened.
* Playback opens the output device at the rate of the first chunk it is handed,
  read from the audio itself rather than from configuration. A change of
  synthesiser therefore cannot silently detune the output - the failure mode
  that produces a chipmunk voice and an hour of debugging.

Quality matters asymmetrically. A poor resampler in front of the recogniser
costs accuracy; the same resampler in front of the speakers merely sounds
slightly worse. soxr is used when installed, and the linear fallback logs a
warning rather than pretending to be equivalent.

## The speech synthesis adapter seam

Speech synthesis is the one layer whose upstream Python API cannot be relied on:
packages get renamed, classes gain streaming methods, signatures change. Rather
than guessing at one shape and hard-coding it, `voxlab.providers.tts.engine`
defines what an engine is - text in, samples out - and offers three ways to
obtain one, most explicit first: a factory named in the configuration, a
transformers-based engine, or a package entry point tried against a few
plausible call shapes.

The escape hatch is the important one. When discovery fails, the error names
`voxlab tts-probe` and the configuration key to set, so a mismatch is a
fifteen-line adapter rather than a fork of this repository.

## Latency after phase 1

Synthesis is now chunked by sentence, so a four-sentence answer starts playing
after the first sentence is rendered instead of the last. Sentence splitting
protects German abbreviations, because a break after "z." is audible as a pause
inside a word.

The language model is still drained completely before synthesis begins. That
remains the largest avoidable cost in the loop and is phase 2 work: the model
should hand each finished sentence to the synthesiser while it is still writing
the next. `TurnMetrics` already separates the two, so the improvement will show
up as a change in `llm_total_s` versus `tts_first_chunk_s` rather than as a
claim.
