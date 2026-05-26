# Pitch Editor

A browser-based tool for visualising and editing the pitch contour of audio recordings. Upload a file, inspect the F0 curve alongside the waveform, detect note boundaries automatically, and drag notes up or down to correct pitch — all with real-time audio playback.

---

## Features

- **F0 extraction** — CREPE (via torchcrepe) at 10 ms resolution, displayed as a semitone-labelled pitch contour
- **Note detection** — Spotify Basic Pitch segments the melody into discrete notes with onset, offset, and MIDI pitch
- **Interactive pitch editing** — drag any note box vertically on the pitch plot to shift it; changes are applied with WORLD vocoder synthesis
- **Smooth boundaries** — three-layer f0 smoothing (raised cosine weight → Gaussian spread → log-f0 blending) plus audio-domain cross-fade; see [SMOOTHING.md](SMOOTHING.md) for details
- **Waveform display** — amplitude plot synced to the pitch plot; both zoom and pan together
- **Playback** — Web Audio API with play/pause, seek-by-click, volume control, and a live cursor on both plots
- **Undo / Redo** — `Cmd+Z` / `Cmd+Shift+Z`; each edit snapshots the full audio and F0 state
- **Keyboard shortcut** — `Space` to play/pause while hovering over either plot

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python · FastAPI · Uvicorn |
| F0 extraction | torchcrepe (CREPE, PyTorch port) |
| Note detection | Spotify Basic Pitch |
| Pitch synthesis | WORLD vocoder (pyworld) |
| Audio loading | librosa · soundfile |
| Frontend | Vanilla JS · Plotly.js · Web Audio API |

---

## Setup

### 1. Create the conda environment

```bash
conda create -n pitch-editor python=3.11 -y
conda activate pitch-editor
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn python-multipart \
            torch torchaudio torchcrepe \
            librosa soundfile scipy \
            basic-pitch pyworld
```

### 3. Run

```bash
./run.sh
```

This starts the server on port 8765 and opens `http://localhost:8765` in your browser automatically. Stop with `Ctrl+C`.

To start manually:

```bash
conda activate pitch-editor
uvicorn main:app --host 0.0.0.0 --port 8765
```

---

## Usage

1. **Upload** — drag and drop an audio file (WAV, MP3, FLAC, OGG, M4A) onto the upload zone or click to browse
2. **Analyse** — click **Analyse**; the backend extracts F0 and detects notes (takes a few seconds)
3. **Inspect** — the waveform and pitch contour appear; orange boxes show detected note boundaries
4. **Play** — click the play button or hover over either plot and press `Space`; click anywhere on the plots to seek
5. **Zoom / pan** — scroll vertically to zoom the time axis; two-finger horizontal swipe to pan; both plots stay in sync
6. **Edit pitch** — hover over an orange note box until the cursor changes to ↕, then drag up or down; the note snaps to semitones
7. **Undo / Redo** — `Cmd+Z` / `Cmd+Shift+Z`

---

## Project structure

```
pitch-editor/
├── main.py          # FastAPI backend — analysis, pitch editing, session management
├── static/
│   └── index.html   # Single-page frontend
├── run.sh           # Start script (activates env, opens browser)
├── SMOOTHING.md     # Technical notes on the boundary smoothing algorithm
└── README.md
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Upload audio; returns F0, waveform, notes, session ID |
| `POST` | `/edit_note` | Pitch-shift a note segment; returns updated audio + F0 |
| `POST` | `/undo` | Revert last edit |
| `POST` | `/redo` | Re-apply last undone edit |
| `GET` | `/` | Serves the frontend |

---

## Notes

- Audio is processed at **22050 Hz** internally. The original file quality is preserved for undo back to the initial state (the original bytes are cached and never passed through the vocoder).
- Sessions are stored in memory; up to 5 sessions are kept (LRU eviction). Re-analysing creates a new session.
- WORLD synthesis is CPU-only. Edits on a 30-second file typically take 2–4 seconds.
