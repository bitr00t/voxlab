# Architecture

## The four seams

Everything that could plausibly be replaced sits behind an interface:

| Seam | Interface | Phase 0 placeholder | Target |
| --- | --- | --- | --- |
| Audio in and out | `AudioTransport` | `null` | local device, then browser |
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
