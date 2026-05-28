import tempfile
import os
import math
import uuid
import base64
import io

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.interpolate import CubicSpline
import librosa
import torchcrepe
import torch
import pyworld as pw
import soundfile as sf
from basic_pitch.inference import predict as bp_predict
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}
MAX_SESSIONS = 5

def _prune():
    while len(_sessions) > MAX_SESSIONS:
        del _sessions[next(iter(_sessions))]


# ── Pitch-shift helper ────────────────────────────────────────────────────────
def _smooth_f0_boundaries(
    f0: np.ndarray,
    n_start_f: int,
    n_end_f: int,
    ramp_f: int,
) -> np.ndarray:
    """
    Smooth f0 at note boundaries after a hard shift using cubic spline interpolation.

    At each boundary, a CubicSpline is fitted through anchor frames on both sides
    (outside the ramp zone), then used to fill in the ramp frames. All work is done
    in log-f0 space so the interpolation is linear in semitones.
    """
    result = f0.copy()
    voiced = f0 > 0
    log_f0 = np.where(voiced, np.log(np.maximum(f0, 1e-8)), 0.0)
    n_context = max(3, ramp_f // 2)  # anchor frames on each side of the ramp

    for ramp_start, ramp_end in [
        (n_start_f - ramp_f, n_start_f),       # left boundary
        (n_end_f,             n_end_f + ramp_f), # right boundary
    ]:
        ramp_start = max(0, ramp_start)
        ramp_end   = min(len(f0) - 1, ramp_end)
        if ramp_end <= ramp_start:
            continue

        # Voiced anchor frames just outside each side of the ramp
        left_ctx  = np.arange(max(0, ramp_start - n_context), ramp_start + 1)
        right_ctx = np.arange(ramp_end, min(len(f0), ramp_end + n_context + 1))
        lv = left_ctx[voiced[left_ctx]]
        rv = right_ctx[voiced[right_ctx]]
        if len(lv) < 2 or len(rv) < 2:
            continue

        xs = np.concatenate([lv, rv]).astype(float)
        ys = log_f0[xs.astype(int)]
        cs = CubicSpline(xs, ys)

        # Fill voiced frames inside the ramp zone with spline values
        inner = np.arange(ramp_start + 1, ramp_end)
        inner_voiced = inner[voiced[inner]]
        if len(inner_voiced) > 0:
            result[inner_voiced] = np.exp(cs(inner_voiced.astype(float)))

    # Low-pass filter the full log-f0 curve around both boundary regions to
    # remove any residual kinks left after spline filling.
    # Unvoiced gaps are filled by interpolation first so their 0.0 log values
    # don't bleed into the Gaussian average for neighbouring voiced frames.
    log_result   = np.where(voiced, np.log(np.maximum(result, 1e-8)), 0.0)
    voiced_idx   = np.where(voiced)[0]
    unvoiced_idx = np.where(~voiced)[0]
    log_filled   = log_result.copy()
    if len(voiced_idx) > 1 and len(unvoiced_idx) > 0:
        log_filled[unvoiced_idx] = np.interp(
            unvoiced_idx.astype(float),
            voiced_idx.astype(float),
            log_result[voiced_idx],
        )
    log_smooth = gaussian_filter1d(log_filled, sigma=ramp_f * 0.4)

    for ramp_start, ramp_end in [
        (n_start_f - ramp_f, n_start_f + ramp_f),
        (n_end_f   - ramp_f, n_end_f   + ramp_f),
    ]:
        ramp_start = max(0, ramp_start)
        ramp_end   = min(len(f0) - 1, ramp_end)
        # Exclude the outermost frame on each side so that the transition
        # anchors exactly to the spline/original value there and the Gaussian
        # average (which reaches into the shifted zone) does not pull the edge
        # frame away from the original pitch.
        zone = np.arange(ramp_start + 1, ramp_end)
        zone_voiced = zone[voiced[zone]]
        if len(zone_voiced) > 0:
            result[zone_voiced] = np.exp(log_smooth[zone_voiced])

    # Moving average centred on each boundary frame to flatten any remaining
    # step at the exact cut point. Window = ramp_f frames; applied in log-f0.
    # Unvoiced gaps are filled by linear interpolation before filtering so that
    # zeros never enter the average window and corrupt nearby voiced frames.
    win = max(3, ramp_f)
    log_result   = np.where(voiced, np.log(np.maximum(result, 1e-8)), 0.0)
    voiced_idx   = np.where(voiced)[0]
    unvoiced_idx = np.where(~voiced)[0]
    log_filled   = log_result.copy()
    if len(voiced_idx) > 1 and len(unvoiced_idx) > 0:
        log_filled[unvoiced_idx] = np.interp(
            unvoiced_idx.astype(float),
            voiced_idx.astype(float),
            log_result[voiced_idx],
        )
    log_ma = uniform_filter1d(log_filled, size=win, mode='reflect')

    for boundary in [n_start_f, n_end_f]:
        half = win // 2
        zone = np.arange(max(0, boundary - half), min(len(f0), boundary + half + 1))
        zone_voiced = zone[voiced[zone]]
        if len(zone_voiced) > 0:
            result[zone_voiced] = np.exp(log_ma[zone_voiced])

    return result


def _splice_with_crossfade(
    audio: np.ndarray,    # Full original audio array (samples, float32).
    synth: np.ndarray,    # Synthesised padded segment: audio[s_start:s_end] re-synthesised.
    s_start: int,         # First sample of the padded segment in `audio`.
    s_end: int,           # One-past-last sample of the padded segment in `audio`.
    left_pad: int,        # Samples of analysis padding before the note start inside `synth`.
    right_pad: int,       # Samples of analysis padding after the note end inside `synth`.
    ramp_s: int,          # Cross-fade half-length in samples (cf_s = ramp_s // 2, capped by padding).
) -> np.ndarray:
    """
    Splice synthesised segment back into audio.

    The entire padded segment [s_start, s_end] is replaced with synthesised audio —
    including the padding zones where WORLD used the original f0. Cross-fades are
    placed at the OUTER edges of the padding (s_start and s_end), not at the note
    boundaries. This avoids pitch beating: in the blend zone both original and
    synthesised audio have the same pitch, so only a phase difference is blended out.
    """
    result  = audio.copy()
    seg_len = s_end - s_start
    cf_s    = min(ramp_s // 2, left_pad, right_pad)

    # Use synthesised audio for the full padded segment
    result[s_start:s_end] = synth[:seg_len]

    # Left crossfade: fade from original → synthesised at the start of the padding.
    # synth[:cf_s] is contiguous with synth[cf_s:] so no waveform gap.
    if cf_s > 0:
        t = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, cf_s))).astype(np.float32)
        result[s_start : s_start + cf_s] = (
            audio[s_start : s_start + cf_s] * (1 - t) + synth[:cf_s] * t
        )

    # Right crossfade: fade from synthesised → original at the end of the padding.
    # synth[off:off+cf_s] is contiguous with synth[:off] so no waveform gap.
    if cf_s > 0:
        t   = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, cf_s))).astype(np.float32)
        off = seg_len - cf_s
        result[s_end - cf_s : s_end] = (
            synth[off : off + cf_s] * t + audio[s_end - cf_s : s_end] * (1 - t)
        )

    return result


def _save_f0_debug(
    f0_orig: np.ndarray,
    f0_shifted: np.ndarray,
    f0_synth: np.ndarray,
    frame_period: float,
    s_start_sec: float = 0.0,
    n_start_f: int = 0,
    n_end_f: int = 0,
    full_f0_times: list | None = None,   # CREPE times for full audio
    full_f0_hz: list | None = None,      # CREPE f0 for full audio (None = unvoiced)
    path: str = "debug_f0.png",
) -> None:
    seg_times = np.arange(len(f0_orig)) * frame_period / 1000.0 + s_start_sec
    note_start_sec = s_start_sec + n_start_f * frame_period / 1000.0
    note_end_sec   = s_start_sec + n_end_f   * frame_period / 1000.0

    def hz_to_midi(hz):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(hz > 0, 69 + 12 * np.log2(hz / 440.0), np.nan)

    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    all_tick_midis  = list(range(36, 97, 1))   # dense pool; we'll filter to visible range
    all_tick_labels = [NOTE_NAMES[m % 12] + str(m // 12 - 1) for m in all_tick_midis]

    titles   = ["Original f0", "f0 after smooth", "f0 after synthesis (re-analysed)"]
    seg_data = [f0_orig, f0_shifted, f0_synth]
    colors   = ["steelblue", "darkorange", "seagreen"]

    # Compute voiced MIDI range across all data to auto-zoom the y-axis.
    all_midi_voiced = []
    for seg_f0 in seg_data:
        v = seg_f0[seg_f0 > 0]
        if len(v):
            all_midi_voiced.extend(hz_to_midi(v).tolist())
    if full_f0_hz is not None:
        fhz = np.array([v for v in full_f0_hz if v is not None and v > 0])
        if len(fhz):
            all_midi_voiced.extend(hz_to_midi(fhz).tolist())
    if all_midi_voiced:
        ylo = min(all_midi_voiced) - 4
        yhi = max(all_midi_voiced) + 4
    else:
        ylo, yhi = 48, 84   # fallback: C3–C6

    # Only show tick marks inside the zoom window, spaced every 3 semitones.
    vis_pairs = [(m, l) for m, l in zip(all_tick_midis, all_tick_labels)
                 if ylo <= m <= yhi and m % 3 == 0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, sharey=True)

    for ax, title, seg_f0, color in zip(axes, titles, seg_data, colors):
        # Full-audio CREPE f0 as grey background reference
        if full_f0_times is not None and full_f0_hz is not None:
            ft = np.array(full_f0_times)
            fh = hz_to_midi(np.array([v if v is not None else 0.0 for v in full_f0_hz]))
            ax.plot(ft, fh, color="#444", linewidth=0.8, alpha=0.5, label="CREPE (full)")

        # Segment f0 (WORLD)
        n = min(len(seg_times), len(seg_f0))
        seg_midi = hz_to_midi(seg_f0[:n])
        ax.plot(seg_times[:n], seg_midi, color=color, linewidth=1.5, label="WORLD (segment)")

        ax.axvline(note_start_sec, color="red",   linewidth=1.2, linestyle="--", label="note start")
        ax.axvline(note_end_sec,   color="tomato", linewidth=1.2, linestyle=":",  label="note end")
        ax.set_ylabel("Semitone")
        ax.set_ylim(ylo, yhi)
        if vis_pairs:
            ax.set_yticks([p[0] for p in vis_pairs])
            ax.set_yticklabels([p[1] for p in vis_pairs], fontsize=7)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pitch_shift_segment(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    semitone_delta: int,
    pad_sec: float  = 0.100,
    ramp_sec: float = 0.080,
    full_f0_times: list | None = None,
    full_f0_hz: list | None = None,
) -> np.ndarray:
    """Pitch-shift audio[start_sec:end_sec] by semitone_delta semitones."""
    if semitone_delta == 0:
        return audio

    pad_s  = int(pad_sec  * sr)
    ramp_s = int(ramp_sec * sr)
    frame_period = 5.0

    s_start   = max(0, int(start_sec * sr) - pad_s)
    s_end     = min(len(audio), int(end_sec * sr) + pad_s)
    left_pad  = int(start_sec * sr) - s_start
    right_pad = s_end - int(end_sec * sr)

    segment = audio[s_start:s_end].astype(np.float64)
    f0, sp, ap = pw.wav2world(segment, float(sr), frame_period=frame_period)

    n_frames  = len(f0)
    hop_s     = frame_period / 1000.0 * sr
    n_start_f = int(left_pad / hop_s)
    n_end_f   = min(int((left_pad + int((end_sec - start_sec) * sr)) / hop_s), n_frames - 1)
    ramp_f    = max(2, int(ramp_s / hop_s))

    # Hard shift: multiply all voiced frames inside the note region
    ratio       = 2.0 ** (semitone_delta / 12.0)
    modified_f0 = f0.copy()
    voiced      = f0 > 0
    in_note     = (np.arange(n_frames) >= n_start_f) & (np.arange(n_frames) <= n_end_f)
    modified_f0[voiced & in_note] *= ratio

    f0_after_hard_shift = modified_f0.copy()

    modified_f0 = _smooth_f0_boundaries(modified_f0, n_start_f, n_end_f, ramp_f)

    f0_after_smooth = modified_f0.copy()

    # Synthesise
    synth = pw.synthesize(modified_f0, sp, ap, float(sr), frame_period=frame_period)
    synth = synth[: len(segment)].astype(np.float32)

    # Re-analyse synthesised audio to get actual output f0
    f0_synth, _, _ = pw.wav2world(synth.astype(np.float64), float(sr), frame_period=frame_period)

    # Save all f0 stages for debugging
    times = np.arange(len(f0)) * frame_period / 1000.0
    np.save("debug_f0_original.npy",        np.stack([times, f0],                  axis=1))
    np.save("debug_f0_after_hard_shift.npy", np.stack([times, f0_after_hard_shift], axis=1))
    np.save("debug_f0_after_smooth.npy",     np.stack([times, f0_after_smooth],     axis=1))
    n = min(len(times), len(f0_synth))
    np.save("debug_f0_after_synth.npy",      np.stack([times[:n], f0_synth[:n]],    axis=1))
    np.save("debug_meta.npy", np.array([n_start_f, n_end_f, ramp_f, frame_period]))

    _save_f0_debug(
        f0, f0_after_smooth, f0_synth, frame_period,
        s_start_sec=s_start / sr,
        n_start_f=n_start_f,
        n_end_f=n_end_f,
        full_f0_times=full_f0_times,
        full_f0_hz=full_f0_hz,
    )

    # Absolute timestamps for each WORLD frame (used to update display f0)
    seg_times_abs = np.arange(n_frames) * frame_period / 1000.0 + s_start / sr

    return (
        _splice_with_crossfade(audio, synth, s_start, s_end, left_pad, right_pad, ramp_s),
        seg_times_abs,
        f0_after_smooth,
        start_sec,
        end_sec,
    )


def _update_display_f0(
    f0_hz: list,
    f0_times: list,
    seg_times: np.ndarray,
    seg_f0_smooth: np.ndarray,
    note_t0: float | None = None,
    note_t1: float | None = None,
) -> list:
    """
    Replace CREPE f0_hz values inside the edited note region with values
    interpolated from the smoothed WORLD f0, so the displayed contour matches
    the synthesised audio.

    note_t0 / note_t1 bound the update zone (note ± ramp). Restricting to the
    note region prevents the padding zone — where WORLD and CREPE may differ
    slightly — from creating a visible step in the display outside the note.
    """
    voiced = seg_f0_smooth > 0
    if voiced.sum() < 2:
        return list(f0_hz)

    vt     = seg_times[voiced]
    log_vf = np.log(seg_f0_smooth[voiced])

    # Fall back to the full segment range if boundaries were not supplied
    update_t0 = note_t0 if note_t0 is not None else seg_times[0]
    update_t1 = note_t1 if note_t1 is not None else seg_times[-1]

    result = list(f0_hz)
    for i, t in enumerate(f0_times):
        if update_t0 <= t <= update_t1 and result[i] is not None:
            result[i] = round(float(np.exp(np.interp(t, vt, log_vf))), 3)
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 22050 Hz for WORLD synthesis and playback — much better voice quality than 16 kHz.
        # CREPE gets its own 16 kHz downsample; torchcrepe resamples internally anyway.
        SR_PLAY  = 22050
        SR_CREPE = 16000
        audio_play  = librosa.load(tmp_path, sr=SR_PLAY,  mono=True)[0]
        audio_crepe = librosa.resample(audio_play, orig_sr=SR_PLAY, target_sr=SR_CREPE)

        # Waveform (downsampled for display)
        hop = max(1, len(audio_play) // 4000)
        waveform       = audio_play[::hop].tolist()
        waveform_times = (np.arange(len(waveform)) * hop / SR_PLAY).tolist()

        # F0 with torchcrepe at 16 kHz
        audio_tensor = torch.from_numpy(audio_crepe).unsqueeze(0)
        hop_length_crepe = int(SR_CREPE * 0.01)          # 10 ms hop at 16 kHz
        hop_length_play  = int(SR_PLAY  * 0.01)          # matching hop at 22050 Hz
        f0, confidence = torchcrepe.predict(
            audio_tensor, SR_CREPE,
            hop_length=hop_length_crepe, fmin=50, fmax=2000,
            model="tiny", batch_size=512, device="cpu",
            return_periodicity=True,
        )
        f0         = f0.squeeze().numpy()
        confidence = confidence.squeeze().numpy()
        # Times are the same regardless of which sr we used for CREPE
        times = np.arange(len(f0)) * hop_length_crepe / SR_CREPE

        voiced_mask = confidence > 0.5
        f0_hz = [float(v) if voiced_mask[i] else None for i, v in enumerate(f0)]

        # Note detection with Basic Pitch
        _, _, note_events = bp_predict(tmp_path)
        notes = sorted(
            [
                {
                    "start":     float(n[0]),
                    "end":       float(n[1]),
                    "midi":      int(n[2]),
                    "hz":        round(440.0 * math.pow(2.0, (int(n[2]) - 69) / 12.0), 3),
                    "amplitude": round(float(n[3]), 3),
                }
                for n in note_events
            ],
            key=lambda n: n["start"],
        )

        # Encode original audio once — reused verbatim when undo returns to initial state
        orig_buf = io.BytesIO()
        sf.write(orig_buf, audio_play, SR_PLAY, format="WAV", subtype="PCM_16")
        orig_audio_b64 = base64.b64encode(orig_buf.getvalue()).decode()

        # Store session
        session_id = str(uuid.uuid4())
        _prune()
        _sessions[session_id] = {
            "audio":          audio_play,       # 22050 Hz, current (mutable)
            "orig_audio_b64": orig_audio_b64,   # original encoded once, never re-encoded
            "sr":             SR_PLAY,
            "hop_length":     hop_length_play,
            "f0_times":       times.tolist(),
            "f0_hz":          f0_hz,
            "notes":          notes,
            "undo_stack":     [],
            "redo_stack":     [],
        }

        return JSONResponse({
            "session_id":  session_id,
            "duration":    float(len(audio_play) / SR_PLAY),
            "sample_rate": SR_PLAY,
            "waveform":    {"times": waveform_times, "amplitudes": waveform},
            "f0":          {"times": times.tolist(), "hz": f0_hz, "confidence": confidence.tolist()},
            "notes":       notes,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


class EditNoteRequest(BaseModel):
    session_id:     str
    note_start:     float
    note_end:       float
    semitone_delta: int


def _session_snapshot(session: dict) -> dict:
    return {
        "audio":     session["audio"].copy(),
        "f0_hz":     list(session["f0_hz"]),
        "notes":     [dict(n) for n in session["notes"]],
        # Flag: True only for the very first snapshot (original audio)
        "is_origin": len(session["undo_stack"]) == 0,
    }

def _audio_response(audio: np.ndarray, sr: int, f0_times: list, f0_hz: list, notes: list) -> dict:
    hop            = max(1, len(audio) // 4000)
    waveform       = audio[::hop].tolist()
    waveform_times = (np.arange(len(waveform)) * hop / sr).tolist()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return {
        "audio_b64": base64.b64encode(buf.getvalue()).decode(),
        "waveform":  {"times": waveform_times, "amplitudes": waveform},
        "f0":        {"times": f0_times, "hz": f0_hz},
        "notes":     notes,
        "can_undo":  True,
    }


@app.post("/edit_note")
async def edit_note(req: EditNoteRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired — please re-analyse.")

    # Snapshot before mutating; new edit invalidates redo history
    session["undo_stack"].append(_session_snapshot(session))
    session["redo_stack"].clear()

    audio    = session["audio"]
    sr       = session["sr"]
    f0_hz    = list(session["f0_hz"])
    f0_times = session["f0_times"]

    # Pitch-shift the segment with smooth boundaries
    modified, seg_times, seg_f0_smooth, note_t0, note_t1 = pitch_shift_segment(
        audio, sr, req.note_start, req.note_end, req.semitone_delta,
        full_f0_times=f0_times,
        full_f0_hz=f0_hz,
    )
    session["audio"] = modified

    # Update displayed f0 only within the note region (± ramp), so that the
    # padding zone — where WORLD and CREPE f0 may differ — is not overwritten.
    ramp_sec = 0.080
    f0_hz = _update_display_f0(
        f0_hz, f0_times, seg_times, seg_f0_smooth,
        note_t0=note_t0 - ramp_sec,
        note_t1=note_t1 + ramp_sec,
    )
    session["f0_hz"] = f0_hz

    # Update note pitch record
    notes = [dict(n) for n in session["notes"]]
    for n in notes:
        if abs(n["start"] - req.note_start) < 0.001 and abs(n["end"] - req.note_end) < 0.001:
            new_midi = n["midi"] + req.semitone_delta
            n["midi"] = new_midi
            n["hz"]   = round(440.0 * math.pow(2.0, (new_midi - 69) / 12.0), 3)
            break
    session["notes"] = notes

    return JSONResponse(_audio_response(modified, sr, f0_times, f0_hz, notes))


class DeleteNoteRequest(BaseModel):
    session_id: str
    note_start:  float
    note_end:    float


@app.post("/delete_note")
async def delete_note(req: DeleteNoteRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired — please re-analyse.")

    session["undo_stack"].append(_session_snapshot(session))
    session["redo_stack"].clear()

    notes = [n for n in session["notes"]
             if not (abs(n["start"] - req.note_start) < 0.001
                     and abs(n["end"] - req.note_end) < 0.001)]
    session["notes"] = notes

    return JSONResponse(_audio_response(
        session["audio"], session["sr"], session["f0_times"], session["f0_hz"], notes
    ))


class MergeNoteRequest(BaseModel):
    session_id: str
    note_start:  float
    note_end:    float


@app.post("/merge_note")
async def merge_note(req: MergeNoteRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired — please re-analyse.")

    notes_sorted = sorted(session["notes"], key=lambda n: n["start"])
    curr_idx = next(
        (i for i, n in enumerate(notes_sorted)
         if abs(n["start"] - req.note_start) < 0.001
         and abs(n["end"] - req.note_end) < 0.001),
        -1,
    )
    if curr_idx <= 0:
        raise HTTPException(status_code=400, detail="No previous note to merge with.")

    session["undo_stack"].append(_session_snapshot(session))
    session["redo_stack"].clear()

    curr_note = notes_sorted[curr_idx]

    new_notes = []
    for i, n in enumerate(notes_sorted):
        if i == curr_idx - 1:
            new_notes.append({**n, "end": curr_note["end"]})
        elif i == curr_idx:
            pass  # absorbed into previous note
        else:
            new_notes.append(dict(n))
    session["notes"] = new_notes

    return JSONResponse(_audio_response(
        session["audio"], session["sr"], session["f0_times"], session["f0_hz"], new_notes
    ))


class ResizeNoteRequest(BaseModel):
    session_id: str
    note_start: float
    note_end:   float
    new_start:  float
    new_end:    float


@app.post("/resize_note")
async def resize_note(req: ResizeNoteRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired — please re-analyse.")

    session["undo_stack"].append(_session_snapshot(session))
    session["redo_stack"].clear()

    notes = [dict(n) for n in session["notes"]]
    for n in notes:
        if abs(n["start"] - req.note_start) < 0.001 and abs(n["end"] - req.note_end) < 0.001:
            n["start"] = round(req.new_start, 4)
            n["end"]   = round(req.new_end, 4)
            break
    session["notes"] = notes

    return JSONResponse(_audio_response(
        session["audio"], session["sr"], session["f0_times"], session["f0_hz"], notes
    ))


class UndoRequest(BaseModel):
    session_id: str


@app.post("/undo")
async def undo(req: UndoRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired — please re-analyse.")
    if not session["undo_stack"]:
        raise HTTPException(status_code=400, detail="Nothing to undo.")

    # Push current state to redo stack before reverting
    session["redo_stack"].append(_session_snapshot(session))

    state = session["undo_stack"].pop()
    session["audio"] = state["audio"]
    session["f0_hz"] = state["f0_hz"]
    session["notes"] = state["notes"]

    resp = _audio_response(
        state["audio"], session["sr"],
        session["f0_times"], state["f0_hz"], state["notes"],
    )
    if state.get("is_origin"):
        resp["audio_b64"] = session["orig_audio_b64"]
    resp["can_undo"] = len(session["undo_stack"]) > 0
    resp["can_redo"] = True
    return JSONResponse(resp)


class RedoRequest(BaseModel):
    session_id: str


@app.post("/redo")
async def redo(req: RedoRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired — please re-analyse.")
    if not session["redo_stack"]:
        raise HTTPException(status_code=400, detail="Nothing to redo.")

    # Push current state to undo stack before advancing
    session["undo_stack"].append(_session_snapshot(session))

    state = session["redo_stack"].pop()
    session["audio"] = state["audio"]
    session["f0_hz"] = state["f0_hz"]
    session["notes"] = state["notes"]

    resp = _audio_response(
        state["audio"], session["sr"],
        session["f0_times"], state["f0_hz"], state["notes"],
    )
    resp["can_undo"] = True
    resp["can_redo"] = len(session["redo_stack"]) > 0
    return JSONResponse(resp)


class ExportRequest(BaseModel):
    session_id: str
    filename:   str = "export.wav"


@app.post("/export")
async def export_audio(req: ExportRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired — please re-analyse.")

    audio = session["audio"]
    sr    = session["sr"]

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    safe_name = req.filename if req.filename.lower().endswith(".wav") else req.filename + ".wav"
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
