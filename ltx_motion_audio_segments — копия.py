import copy
import glob
import json
import logging
import math
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import tempfile
import uuid
import wave

import folder_paths
import server
import torch
from comfy.cli_args import args
from comfy_api.latest import Types
from comfy_api.latest._input_impl.video_types import VideoFromComponents
from aiohttp import web
from PIL import Image, ImageOps

from comfy_extras.nodes_audio import load as load_audio_file


MAX_STORYBOARD_SLOTS = 32


def _strip_path(path):
    path = (path or "").strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def _parse_timecode(value):
    if value is None:
        raise ValueError("Time value is missing")

    if isinstance(value, (int, float)):
        return max(0.0, float(value))

    text = str(value).strip()
    if not text:
        raise ValueError("Time value is empty")

    if text.replace(".", "", 1).isdigit():
        return max(0.0, float(text))

    parts = [part.strip() for part in text.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"Unsupported time format: {value}")

    try:
        numeric = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Unsupported time format: {value}") from exc

    if len(numeric) == 2:
        minutes, seconds = numeric
        total_seconds = (minutes * 60.0) + seconds
    else:
        hours, minutes, seconds = numeric
        total_seconds = (hours * 3600.0) + (minutes * 60.0) + seconds

    return max(0.0, total_seconds)


def _parse_optional_timecode(value, default_seconds):
    text = "" if value is None else str(value).strip()
    if not text:
        return float(default_seconds)
    return _parse_timecode(text)


def _parse_duration_list(value):
    text = "" if value is None else str(value).strip()
    if not text:
        return []

    parts = text.replace("\r", "\n").replace(",", "\n").split("\n")
    durations = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        duration = float(cleaned)
        if duration <= 0.0:
            raise ValueError("All durations in durations_csv must be greater than 0")
        durations.append(duration)

    return durations


def _parse_keyframe_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        text = str(value).strip()
        if not text:
            return []
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            raw_values = parsed.get("keyframes", [])
        else:
            raw_values = parsed

    keyframes = []
    for item in raw_values:
        seconds = max(0.0, float(item))
        keyframes.append(seconds)
    return keyframes


def _normalize_keyframe_list(keyframes, total_duration=None):
    normalized = []
    seen = set()
    for item in keyframes or []:
        seconds = max(0.0, float(item))
        if total_duration is not None:
            upper_bound = max(0.0, float(total_duration) - 0.001)
            seconds = min(seconds, upper_bound)
        bucket = int(round(seconds * 1000.0))
        if bucket in seen:
            continue
        seen.add(bucket)
        normalized.append(seconds)

    normalized.sort()
    if not normalized or normalized[0] > 0.001:
        normalized.insert(0, 0.0)
    return normalized


def _parse_line_list(value):
    text = "" if value is None else str(value).strip()
    if not text:
        return []

    entries = []
    for line in text.replace("\r", "\n").split("\n"):
        cleaned = line.strip().strip('"')
        if cleaned:
            entries.append(cleaned)
    return entries


def _extract_indexed_kwargs(kwargs, prefix):
    indexed_values = {}
    for key, value in kwargs.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix.isdigit():
            continue
        indexed_values[int(suffix)] = value
    return indexed_values


def _list_input_audio_files():
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    discovered = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            extension = os.path.splitext(filename)[1].lower()
            if extension not in audio_extensions:
                continue
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, input_dir).replace("\\", "/")
            discovered.append(relative_path)

    return sorted(discovered)


class _ContainsAnyOptionalDict(dict):
    def __contains__(self, key):
        return True


def _normalize_audio_tensor(audio):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate


def _slice_audio(audio, start_frame, end_frame):
    waveform, sample_rate = _normalize_audio_tensor(audio)
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame + 1, int(end_frame))
    return {
        "waveform": waveform[..., start_frame:end_frame],
        "sample_rate": sample_rate,
    }


def _resolve_audio_path(audio_file):
    audio_file = _strip_path(audio_file)
    if not audio_file:
        raise ValueError("audio_file is empty")

    if os.path.isabs(audio_file) and os.path.isfile(audio_file):
        return audio_file

    try:
        annotated = folder_paths.get_annotated_filepath(audio_file)
        if annotated and os.path.isfile(annotated):
            return annotated
    except Exception:
        pass

    input_candidate = os.path.join(folder_paths.get_input_directory(), audio_file)
    if os.path.isfile(input_candidate):
        return input_candidate

    raise ValueError(f"Audio file not found: {audio_file}")


def _resolve_image_path(image_file):
    image_file = _strip_path(image_file)
    if not image_file:
        raise ValueError("image_file is empty")

    if os.path.isabs(image_file) and os.path.isfile(image_file):
        return image_file

    try:
        annotated = folder_paths.get_annotated_filepath(image_file)
        if annotated and os.path.isfile(annotated):
            return annotated
    except Exception:
        pass

    input_candidate = os.path.join(folder_paths.get_input_directory(), image_file)
    if os.path.isfile(input_candidate):
        return input_candidate

    raise ValueError(f"Image file not found: {image_file}")


def _load_image_tensor(image_file):
    image_path = _resolve_image_path(image_file)
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None, ...]
    return image_tensor


def _read_waveform_peaks(audio_file, bins=1200):
    audio_path = _resolve_audio_path(audio_file)
    extension = os.path.splitext(audio_path)[1].lower()
    if extension != ".wav":
        raise ValueError("Waveform preview currently supports WAV files")

    with wave.open(audio_path, "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frame_bytes = wav_file.readframes(frame_count)

    if frame_count <= 0 or sample_rate <= 0:
        return {
            "duration": 0.0,
            "sample_rate": sample_rate,
            "peaks": [],
            "audio_path": audio_path,
        }

    if sample_width == 1:
        samples = np.frombuffer(frame_bytes, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        signed = (signed ^ sign_mask) - sign_mask
        samples = signed.astype(np.float32) / float(1 << 23)
    elif sample_width == 4:
        samples = np.frombuffer(frame_bytes, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channel_count > 1:
        samples = samples.reshape(-1, channel_count)
        samples = np.mean(np.abs(samples), axis=1)
    else:
        samples = np.abs(samples)

    bins = max(64, min(int(bins), 4096))
    if samples.size == 0:
        peaks = []
    else:
        edges = np.linspace(0, samples.size, num=bins + 1, dtype=np.int64)
        peaks = []
        for index in range(bins):
            start = edges[index]
            end = edges[index + 1]
            if end <= start:
                peaks.append(0.0)
                continue
            peaks.append(float(np.max(samples[start:end])))

    duration = float(frame_count) / float(sample_rate)
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "peaks": peaks,
        "audio_path": audio_path,
    }


def _select_storyboard_segment(audio, items, durations, start_time="", end_time="", segment_index=0):
    if not items:
        raise ValueError("At least one storyboard image is required")

    waveform, sample_rate = _normalize_audio_tensor(audio)
    if not sample_rate:
        raise ValueError("Audio sample_rate is missing or zero")

    total_frames = waveform.shape[-1]
    total_duration = total_frames / sample_rate
    window_start_seconds = _parse_optional_timecode(start_time, 0.0)
    window_end_seconds = _parse_optional_timecode(end_time, total_duration)
    window_start_seconds = min(max(0.0, window_start_seconds), total_duration)
    window_end_seconds = min(max(window_start_seconds, window_end_seconds), total_duration)
    if window_end_seconds <= window_start_seconds:
        raise ValueError("Selected audio window is empty")

    window_start_frame = int(math.floor(window_start_seconds * sample_rate))
    window_end_frame = int(math.ceil(window_end_seconds * sample_rate))

    scheduled_items = list(items)
    scheduled_durations = []
    for index in range(len(scheduled_items)):
        duration_value = durations[min(index, len(durations) - 1)]
        scheduled_durations.append(max(0.1, float(duration_value)))

    segments = []
    cursor = window_start_frame
    slot_index = 0
    last_duration = scheduled_durations[-1]
    last_item = scheduled_items[-1]
    while cursor < window_end_frame:
        active_slot = min(slot_index, len(scheduled_items) - 1)
        active_item = scheduled_items[active_slot] if slot_index < len(scheduled_items) else last_item
        active_duration = scheduled_durations[active_slot] if slot_index < len(scheduled_items) else last_duration
        segment_frames = max(1, int(round(active_duration * sample_rate)))
        next_cursor = min(window_end_frame, cursor + segment_frames)
        segments.append((cursor, next_cursor, active_item, active_duration, active_slot + 1, slot_index >= len(scheduled_items)))
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        slot_index += 1

    total_segments = max(1, len(segments))
    current_segment = min(max(int(segment_index), 0), total_segments - 1)
    segment_start, segment_end, selected_item, _scheduled_duration, selected_slot, reused_last = segments[current_segment]
    selected_audio = _slice_audio(audio, segment_start, segment_end)
    segment_duration_seconds = (segment_end - segment_start) / sample_rate
    debug_text = (
        f"segment={current_segment + 1}/{total_segments} | slot={selected_slot} | reused_last={reused_last} | "
        f"window={window_start_seconds:.3f}s->{window_end_seconds:.3f}s | "
        f"segment={segment_start / sample_rate:.3f}s->{segment_end / sample_rate:.3f}s | "
        f"duration={segment_duration_seconds:.3f}s | image_count={len(scheduled_items)}"
    )
    return selected_audio, selected_item, current_segment, total_segments, float(segment_duration_seconds), debug_text


def _select_keyframed_storyboard_segment(audio, items, keyframes, segment_index=0, max_segment_seconds=30.0):
    if not items:
        raise ValueError("At least one storyboard image is required")

    waveform, sample_rate = _normalize_audio_tensor(audio)
    if not sample_rate:
        raise ValueError("Audio sample_rate is missing or zero")

    total_frames = waveform.shape[-1]
    total_duration = total_frames / sample_rate
    starts = _normalize_keyframe_list(keyframes, total_duration=total_duration)
    total_segments = max(1, len(starts))
    current_segment = min(max(int(segment_index), 0), total_segments - 1)

    start_seconds = starts[current_segment]
    if current_segment + 1 < len(starts):
        natural_end_seconds = starts[current_segment + 1]
    else:
        natural_end_seconds = total_duration

    clamped_end_seconds = min(natural_end_seconds, start_seconds + max(0.1, float(max_segment_seconds)), total_duration)
    if clamped_end_seconds <= start_seconds:
        clamped_end_seconds = min(total_duration, start_seconds + max(1.0 / sample_rate, 0.1))

    start_frame = int(math.floor(start_seconds * sample_rate))
    end_frame = int(math.ceil(clamped_end_seconds * sample_rate))
    selected_audio = _slice_audio(audio, start_frame, end_frame)
    selected_item = items[min(current_segment, len(items) - 1)]
    segment_duration_seconds = max(0.1, clamped_end_seconds - start_seconds)
    was_clamped = clamped_end_seconds + 1e-6 < natural_end_seconds
    debug_text = (
        f"segment={current_segment + 1}/{total_segments} | keyframe={start_seconds:.3f}s | "
        f"next={natural_end_seconds:.3f}s | duration={segment_duration_seconds:.3f}s | "
        f"clamped_to_30s={was_clamped}"
    )
    return selected_audio, selected_item, current_segment, total_segments, float(segment_duration_seconds), debug_text


async def ltx23_motion_audio_file(request):
    audio_file = request.query.get("audio_file", "")
    try:
        audio_path = _resolve_audio_path(audio_file)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    return web.FileResponse(audio_path, headers={"Content-Type": content_type})


async def ltx23_motion_audio_waveform(request):
    audio_file = request.query.get("audio_file", "")
    bins = request.query.get("bins", "1200")
    try:
        payload = _read_waveform_peaks(audio_file, bins=int(bins))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    payload["audio_url"] = f"/ltx23-motion/audio-file?audio_file={audio_file}"
    return web.json_response(payload)


_prompt_server_instance = getattr(server.PromptServer, "instance", None)
if _prompt_server_instance is not None:
    _prompt_server_instance.routes.get("/ltx23-motion/audio-file")(ltx23_motion_audio_file)
    _prompt_server_instance.routes.get("/ltx23-motion/audio-waveform")(ltx23_motion_audio_waveform)


def _get_current_queue_item():
    prompt_queue = server.PromptServer.instance.prompt_queue
    current = next(iter(prompt_queue.currently_running.values()))

    if len(current) == 6:
        (_, _, prompt, extra_data, outputs_to_execute, sensitive) = current
    else:
        (_, _, prompt, extra_data, outputs_to_execute) = current
        sensitive = {}
    return prompt, extra_data, outputs_to_execute, sensitive


def _normalize_outputs_to_execute(outputs_to_execute):
    if outputs_to_execute is None:
        return None
    return [str(item) for item in outputs_to_execute]


def _normalize_prompt_keys(prompt):
    if prompt is None:
        return None
    return {str(key): value for key, value in prompt.items()}


def _prompt_node_snapshot(prompt, node_id):
    if not isinstance(prompt, dict):
        return None
    node = prompt.get(str(node_id)) or prompt.get(node_id)
    if not isinstance(node, dict):
        return None
    return {
        "class_type": node.get("class_type"),
        "inputs": node.get("inputs"),
        "is_output": node.get("is_output"),
        "meta": node.get("_meta"),
    }


def _log_prompt_cycle_excerpt(prefix, prompt):
    if not isinstance(prompt, dict):
        logging.info("%s prompt unavailable: %s", prefix, type(prompt).__name__)
        return

    target_nodes = ("350", "351", "352", "361", "362", "365", "377", "396", "399", "400")
    excerpt = {node_id: _prompt_node_snapshot(prompt, node_id) for node_id in target_nodes}
    logging.info("%s prompt excerpt: %s", prefix, excerpt)


def _enqueue_prompt(prompt):
    prompt_queue = server.PromptServer.instance.prompt_queue
    _, extra_data, outputs_to_execute, sensitive = _get_current_queue_item()
    prompt = _normalize_prompt_keys(prompt)
    outputs_to_execute = _normalize_outputs_to_execute(outputs_to_execute)

    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    prompt_id = str(server.uuid.uuid4())
    logging.info("[LTX Motion] Queueing next segment prompt %s with outputs %s", prompt_id, outputs_to_execute)
    prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive))


def _find_ffmpeg():
    forced = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if forced and os.path.isfile(forced):
        return forced

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_path = get_ffmpeg_exe()
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            return ffmpeg_path
    except Exception:
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    for candidate in ("ffmpeg.exe", "ffmpeg"):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def _build_metadata(prompt, extra_pnginfo):
    if args.disable_metadata:
        return None

    metadata = {}
    if extra_pnginfo is not None:
        metadata.update(extra_pnginfo)
    if prompt is not None:
        metadata["prompt"] = prompt
    return metadata or None


def _resolve_output_location(filename_prefix, video):
    width, height = video.get_dimensions()
    full_output_folder, filename, counter, subfolder, _resolved_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
        width,
        height,
    )

    os.makedirs(full_output_folder, exist_ok=True)
    next_counter = max(int(counter or 1), 1)
    while True:
        unique_base_name = f"{filename}_{next_counter:05d}"
        existing_matches = glob.glob(os.path.join(full_output_folder, f"{unique_base_name}*"))
        if not existing_matches:
            break
        next_counter += 1

    logging.info(
        "[LTX Motion] Resolved output base name: folder=%s base=%s subfolder=%s start_counter=%s final_counter=%s",
        full_output_folder,
        unique_base_name,
        subfolder,
        counter,
        next_counter,
    )
    return full_output_folder, unique_base_name, subfolder


def _video_extension():
    return Types.VideoContainer.get_extension("auto")


def _segment_filename(base_name, render_id, segment_index):
    return f"{base_name}_{render_id}_seg_{segment_index + 1:04d}.{_video_extension()}"


def _final_filename(base_name, render_id):
    return f"{base_name}_{render_id}_full.{_video_extension()}"


def _concat_segments(segment_paths, output_path):
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg was not found, so segment videos could not be concatenated.")

    missing_paths = [segment_path for segment_path in segment_paths if not os.path.exists(segment_path)]
    if missing_paths:
        missing_lines = "\n".join(missing_paths)
        raise RuntimeError(f"Segment merge aborted because these files are missing:\n{missing_lines}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = handle.name
        for segment_path in segment_paths:
            escaped = segment_path.replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    primary_command = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        # Copy video to avoid a full re-render, but decode/re-encode audio so
        # each segment's codec padding does not create audible gaps at joins.
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        output_path,
    ]
    fallback_command = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        subprocess.run(primary_command, check=True, capture_output=True)
    except subprocess.CalledProcessError as primary_exc:
        logging.warning(
            "[LTX Motion] Audio-safe concat with copied video failed, retrying with full re-encode: %s",
            primary_exc.stderr.decode("utf-8", errors="replace"),
        )
        try:
            subprocess.run(fallback_command, check=True, capture_output=True)
        except subprocess.CalledProcessError as fallback_exc:
            raise RuntimeError(fallback_exc.stderr.decode("utf-8", errors="replace")) from fallback_exc
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def _sanitize_video_audio(video):
    if not hasattr(video, "get_components"):
        return video

    components = video.get_components()
    audio = getattr(components, "audio", None)
    if not audio:
        return video

    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    if waveform is None or torch.isfinite(waveform).all():
        return video

    invalid_samples = int((~torch.isfinite(waveform)).sum().item())
    sanitized_audio = dict(audio)
    sanitized_audio["waveform"] = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    components.audio = sanitized_audio
    logging.warning("[LTX Motion] Sanitized %s invalid audio samples before video save", invalid_samples)
    return VideoFromComponents(components)


def _sanitize_video_images(video):
    if not hasattr(video, "get_components"):
        return video

    components = video.get_components()
    images = getattr(components, "images", None)
    if images is None or torch.isfinite(images).all():
        return video

    invalid_pixels = int((~torch.isfinite(images)).sum().item())
    components.images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
    logging.warning("[LTX Motion] Sanitized %s invalid image pixels before video save", invalid_pixels)
    return VideoFromComponents(components)


def _summarize_video_frames(video):
    if not hasattr(video, "get_components"):
        return "video diagnostics unavailable"

    try:
        components = video.get_components()
        images = getattr(components, "images", None)
        if images is None or getattr(images, "numel", lambda: 0)() == 0:
            return "frames=0"

        invalid_pixels = int((~torch.isfinite(images)).sum().item())
        finite_images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
        min_value = float(finite_images.min().item())
        max_value = float(finite_images.max().item())
        mean_value = float(finite_images.mean().item())
        first_frame_mean = float(finite_images[0].mean().item()) if finite_images.shape[0] else mean_value
        return (
            f"frames={finite_images.shape[0]} | invalid_pixels={invalid_pixels} | "
            f"min={min_value:.6f} | max={max_value:.6f} | mean={mean_value:.6f} | first_frame_mean={first_frame_mean:.6f}"
        )
    except Exception as exc:
        return f"video diagnostics failed: {exc}"


def _summarize_tensor(tensor):
    invalid_values = int((~torch.isfinite(tensor)).sum().item())
    finite_tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        f"shape={tuple(tensor.shape)} | invalid_values={invalid_values} | "
        f"min={float(finite_tensor.min().item()):.6f} | max={float(finite_tensor.max().item()):.6f} | "
        f"mean={float(finite_tensor.mean().item()):.6f}"
    )


class LTXMotionDebugImageStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "label": ("STRING", {"default": "image_debug"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "debug_text")
    FUNCTION = "debug_image"
    CATEGORY = "LTX Motion/debug"

    def debug_image(self, image, label="image_debug"):
        debug_text = f"{label} | {_summarize_tensor(image)}"
        logging.info("[LTX Motion][debug] %s", debug_text)
        return (image, debug_text)


class LTXMotionDebugLatentStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "label": ("STRING", {"default": "latent_debug"}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "debug_text")
    FUNCTION = "debug_latent"
    CATEGORY = "LTX Motion/debug"

    def debug_latent(self, latent, label="latent_debug"):
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None:
            debug_text = f"{label} | latent has no samples"
        else:
            debug_text = f"{label} | {_summarize_tensor(samples)}"
        logging.info("[LTX Motion][debug] %s", debug_text)
        return (latent, debug_text)


def _save_video_output(video, filename_prefix, prompt=None, extra_pnginfo=None):
    filename_prefix = filename_prefix or "video/LTX_2.3_ia2v"
    output_folder, base_name, _subfolder = _resolve_output_location(filename_prefix, video)
    output_name = f"{base_name}.{_video_extension()}"
    output_path = os.path.join(output_folder, output_name)
    metadata = _build_metadata(prompt, extra_pnginfo)
    frame_summary = _summarize_video_frames(video)
    logging.info("[LTX Motion] Final video frame summary before save: %s", frame_summary)
    video = _sanitize_video_images(video)
    video = _sanitize_video_audio(video)
    video.save_to(output_path, format=Types.VideoContainer("auto"), codec="auto", metadata=metadata)
    return output_path, output_name


class LTXMotionSegmentedAudioLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segment_seconds": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 600.0, "step": 0.1}),
            },
            "optional": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "INT")
    RETURN_NAMES = ("audio", "current_segment", "total_segments")
    FUNCTION = "load_segment"
    CATEGORY = "LTX Motion/audio"

    @classmethod
    def VALIDATE_INPUTS(cls, audio=None, segment_seconds=20.0, segment_index=0, input_types=None, **kwargs):
        try:
            audio_info = None
            if isinstance(audio, dict):
                waveform = audio.get("waveform")
                sample_rate = audio.get("sample_rate")
                audio_info = {
                    "waveform_shape": tuple(waveform.shape) if hasattr(waveform, "shape") else type(waveform).__name__,
                    "sample_rate": sample_rate,
                }
            else:
                audio_info = type(audio).__name__
            logging.info(
                "[LTX Motion][validate] Segmented loader inputs: segment_seconds=%s segment_index=%s audio=%s input_types=%s kwargs=%s",
                segment_seconds,
                segment_index,
                audio_info,
                input_types,
                sorted(kwargs.keys()),
            )
        except Exception as exc:
            logging.exception("[LTX Motion][validate] Failed to log segmented loader inputs: %s", exc)
        return True

    def load_segment(self, audio, segment_seconds, segment_index=0):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_frames = waveform.shape[-1]
        safe_segment_seconds = max(float(segment_seconds), 0.01)
        segment_frames = max(1, int(round(safe_segment_seconds * sample_rate)))
        total_segments = max(1, math.ceil(total_frames / segment_frames))
        current_segment = min(max(int(segment_index), 0), total_segments - 1)

        start_frame = current_segment * segment_frames
        end_frame = min(total_frames, start_frame + segment_frames)
        if end_frame <= start_frame:
            end_frame = total_frames

        audio_segment = {
            "waveform": waveform[..., start_frame:end_frame],
            "sample_rate": sample_rate,
        }
        logging.info(
            "[LTX Motion] Loaded audio segment %s/%s samples=%s:%s duration=%.3fs target=%.3fs",
            current_segment + 1,
            total_segments,
            start_frame,
            end_frame,
            (end_frame - start_frame) / sample_rate if sample_rate else 0.0,
            safe_segment_seconds,
        )
        return (audio_segment, current_segment, total_segments)


class LTXMotionAudioRangeExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("STRING", {"default": "1:10", "multiline": False}),
                "end_time": ("STRING", {"default": "1:20", "multiline": False}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "duration_seconds", "debug_text")
    FUNCTION = "extract_range"
    CATEGORY = "LTX Motion/audio"

    @classmethod
    def VALIDATE_INPUTS(cls, audio=None, start_time="1:10", end_time="1:20", input_types=None, **kwargs):
        try:
            start_seconds = _parse_timecode(start_time)
            end_seconds = _parse_timecode(end_time)
            if end_seconds <= start_seconds:
                return "end_time must be greater than start_time"
            logging.info(
                "[LTX Motion][validate] Audio range extractor: start=%s(%.3fs) end=%s(%.3fs) input_types=%s kwargs=%s",
                start_time,
                start_seconds,
                end_time,
                end_seconds,
                input_types,
                sorted(kwargs.keys()),
            )
        except Exception as exc:
            return str(exc)
        return True

    def extract_range(self, audio, start_time, end_time):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        if not sample_rate:
            raise ValueError("Audio sample_rate is missing or zero")

        total_frames = waveform.shape[-1]
        total_duration = total_frames / sample_rate

        start_seconds = _parse_timecode(start_time)
        end_seconds = _parse_timecode(end_time)
        if end_seconds <= start_seconds:
            raise ValueError("end_time must be greater than start_time")

        start_frame = min(total_frames, max(0, int(math.floor(start_seconds * sample_rate))))
        end_frame = min(total_frames, max(start_frame + 1, int(math.ceil(end_seconds * sample_rate))))
        if start_frame >= total_frames:
            raise ValueError(
                f"start_time {start_time} is beyond the audio length of {total_duration:.3f}s"
            )

        audio_segment = {
            "waveform": waveform[..., start_frame:end_frame],
            "sample_rate": sample_rate,
        }
        duration_seconds = (end_frame - start_frame) / sample_rate
        debug_text = (
            f"start={start_time} ({start_seconds:.3f}s) | "
            f"end={end_time} ({end_seconds:.3f}s) | "
            f"duration={duration_seconds:.3f}s | "
            f"samples={start_frame}:{end_frame} | "
            f"total_audio={total_duration:.3f}s"
        )
        logging.info(
            "[LTX Motion] Extracted audio range %s -> %s as samples=%s:%s duration=%.3fs of total %.3fs",
            start_time,
            end_time,
            start_frame,
            end_frame,
            duration_seconds,
            total_duration,
        )
        return (audio_segment, float(duration_seconds), debug_text)


class LTXMotionStoryboardSegmentSelector:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "audio": ("AUDIO",),
            "image_1": ("IMAGE",),
            "image_count": ("INT", {"default": 1, "min": 1, "max": MAX_STORYBOARD_SLOTS, "step": 1}),
            "start_time": ("STRING", {"default": "", "multiline": False}),
            "end_time": ("STRING", {"default": "", "multiline": False}),
            "durations_csv": ("STRING", {"default": "", "multiline": False}),
        }
        for index in range(1, MAX_STORYBOARD_SLOTS + 1):
            default_duration = 5.0 if index == 1 else 10.0
            required[f"duration_{index}"] = ("FLOAT", {"default": default_duration, "min": 0.1, "max": 600.0, "step": 0.1})

        optional = {
            "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            "render_id": ("STRING", {"default": ""}),
        }
        for index in range(2, MAX_STORYBOARD_SLOTS + 1):
            optional[f"image_{index}"] = ("IMAGE",)

        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "image", "current_segment", "total_segments", "segment_duration_seconds", "debug_text")
    FUNCTION = "select_segment"
    CATEGORY = "LTX Motion/audio"

    @classmethod
    def VALIDATE_INPUTS(cls, image_count=1, start_time="", end_time="", durations_csv="", render_id="", input_types=None, **kwargs):
        try:
            image_count = min(max(int(image_count), 1), MAX_STORYBOARD_SLOTS)
            start_seconds = _parse_optional_timecode(start_time, 0.0)
            end_text = "" if end_time is None else str(end_time).strip()
            end_seconds = None if not end_text else _parse_timecode(end_text)
            if end_seconds is not None and end_seconds <= start_seconds:
                return "end_time must be greater than start_time"
            csv_durations = _parse_duration_list(durations_csv)
            duration_limit = max(image_count, MAX_STORYBOARD_SLOTS if not csv_durations else len(csv_durations))
            for index in range(1, duration_limit + 1):
                if csv_durations:
                    duration = csv_durations[min(index - 1, len(csv_durations) - 1)]
                else:
                    fallback_index = min(index, MAX_STORYBOARD_SLOTS)
                    duration = float(kwargs.get(f"duration_{fallback_index}", 0.0))
                if duration <= 0.0:
                    return f"duration_{index} must be greater than 0"
            logging.info(
                "[LTX Motion][validate] Storyboard selector: image_count=%s start=%s end=%s dynamic_images=%s durations_csv_present=%s input_types=%s",
                image_count,
                start_time,
                end_time,
                sorted(_extract_indexed_kwargs(kwargs, "image_")),
                bool(csv_durations),
                input_types,
            )
        except Exception as exc:
            return str(exc)
        return True

    def select_segment(
        self,
        audio,
        image_1,
        image_count,
        start_time="",
        end_time="",
        durations_csv="",
        duration_1=5.0,
        duration_2=10.0,
        duration_3=10.0,
        duration_4=10.0,
        duration_5=10.0,
        duration_6=10.0,
        duration_7=10.0,
        duration_8=10.0,
        duration_9=10.0,
        duration_10=10.0,
        duration_11=10.0,
        duration_12=10.0,
        duration_13=10.0,
        duration_14=10.0,
        duration_15=10.0,
        duration_16=10.0,
        duration_17=10.0,
        duration_18=10.0,
        duration_19=10.0,
        duration_20=10.0,
        duration_21=10.0,
        duration_22=10.0,
        duration_23=10.0,
        duration_24=10.0,
        duration_25=10.0,
        duration_26=10.0,
        duration_27=10.0,
        duration_28=10.0,
        duration_29=10.0,
        duration_30=10.0,
        duration_31=10.0,
        duration_32=10.0,
        segment_index=0,
        render_id="",
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        **kwargs,
    ):
        requested_image_count = min(max(int(image_count), 1), MAX_STORYBOARD_SLOTS)
        durations = [
            float(duration_1),
            float(duration_2),
            float(duration_3),
            float(duration_4),
            float(duration_5),
            float(duration_6),
            float(duration_7),
            float(duration_8),
            float(duration_9),
            float(duration_10),
            float(duration_11),
            float(duration_12),
            float(duration_13),
            float(duration_14),
            float(duration_15),
            float(duration_16),
            float(duration_17),
            float(duration_18),
            float(duration_19),
            float(duration_20),
            float(duration_21),
            float(duration_22),
            float(duration_23),
            float(duration_24),
            float(duration_25),
            float(duration_26),
            float(duration_27),
            float(duration_28),
            float(duration_29),
            float(duration_30),
            float(duration_31),
            float(duration_32),
        ]
        image_values = {
            1: image_1,
            2: image_2,
            3: image_3,
            4: image_4,
            5: image_5,
            6: image_6,
            7: image_7,
            8: image_8,
            9: image_9,
            10: image_10,
        }
        image_values.update(_extract_indexed_kwargs(kwargs, "image_"))

        connected_image_indexes = sorted(index for index, value in image_values.items() if value is not None)
        image_count = requested_image_count
        csv_durations = _parse_duration_list(durations_csv)

        scheduled_images = []
        last_image = image_1
        for index in range(1, image_count + 1):
            image_value = image_values.get(index)
            if image_value is not None:
                last_image = image_value
            scheduled_images.append(last_image)

        scheduled_durations = []
        for index in range(1, image_count + 1):
            if csv_durations:
                duration_value = csv_durations[min(index - 1, len(csv_durations) - 1)]
            else:
                duration_value = durations[min(index - 1, len(durations) - 1)]
            duration_value = max(0.1, float(duration_value))
            scheduled_durations.append(duration_value)
        active_segment_index = int(segment_index) if str(render_id or "").strip() else 0
        selected_audio, selected_image, current_segment, total_segments, segment_duration_seconds, debug_text = _select_storyboard_segment(
            audio,
            scheduled_images,
            scheduled_durations,
            start_time=start_time,
            end_time=end_time,
            segment_index=active_segment_index,
        )
        debug_text = f"{debug_text} | dynamic_images={connected_image_indexes} | render_id={render_id or 'fresh'}"
        logging.info("[LTX Motion] Storyboard selector %s", debug_text)
        return (
            selected_audio,
            selected_image,
            current_segment,
            total_segments,
            float(segment_duration_seconds),
            debug_text,
        )


class LTXMotionStoryboardSegmentPromptSelector:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "audio": ("AUDIO",),
            "image_1": ("IMAGE",),
            "image_count": ("INT", {"default": 1, "min": 1, "max": MAX_STORYBOARD_SLOTS, "step": 1}),
            "start_time": ("STRING", {"default": "", "multiline": False}),
            "end_time": ("STRING", {"default": "", "multiline": False}),
            "durations_csv": ("STRING", {"default": "", "multiline": False}),
        }
        for index in range(1, MAX_STORYBOARD_SLOTS + 1):
            default_duration = 5.0 if index == 1 else 10.0
            required[f"duration_{index}"] = ("FLOAT", {"default": default_duration, "min": 0.1, "max": 600.0, "step": 0.1})
        for index in range(1, MAX_STORYBOARD_SLOTS + 1):
            required[f"prompt_{index}"] = ("STRING", {"default": "", "multiline": True})

        optional = {
            "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            "render_id": ("STRING", {"default": ""}),
        }
        for index in range(2, MAX_STORYBOARD_SLOTS + 1):
            optional[f"image_{index}"] = ("IMAGE",)

        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("audio", "image", "current_segment", "total_segments", "segment_duration_seconds", "selected_prompt", "debug_text")
    FUNCTION = "select_segment"
    CATEGORY = "LTX Motion/audio"

    @classmethod
    def VALIDATE_INPUTS(cls, image_count=1, start_time="", end_time="", durations_csv="", render_id="", input_types=None, **kwargs):
        try:
            image_count = min(max(int(image_count), 1), MAX_STORYBOARD_SLOTS)
            start_seconds = _parse_optional_timecode(start_time, 0.0)
            end_text = "" if end_time is None else str(end_time).strip()
            end_seconds = None if not end_text else _parse_timecode(end_text)
            if end_seconds is not None and end_seconds <= start_seconds:
                return "end_time must be greater than start_time"
            csv_durations = _parse_duration_list(durations_csv)
            duration_limit = max(image_count, MAX_STORYBOARD_SLOTS if not csv_durations else len(csv_durations))
            for index in range(1, duration_limit + 1):
                if csv_durations:
                    duration = csv_durations[min(index - 1, len(csv_durations) - 1)]
                else:
                    fallback_index = min(index, MAX_STORYBOARD_SLOTS)
                    duration = float(kwargs.get(f"duration_{fallback_index}", 0.0))
                if duration <= 0.0:
                    return f"duration_{index} must be greater than 0"
            logging.info(
                "[LTX Motion][validate] Storyboard prompt selector: image_count=%s start=%s end=%s dynamic_images=%s durations_csv_present=%s input_types=%s",
                image_count,
                start_time,
                end_time,
                sorted(_extract_indexed_kwargs(kwargs, "image_")),
                bool(csv_durations),
                input_types,
            )
        except Exception as exc:
            return str(exc)
        return True

    def select_segment(
        self,
        audio,
        image_1,
        image_count,
        start_time="",
        end_time="",
        durations_csv="",
        duration_1=5.0,
        duration_2=10.0,
        duration_3=10.0,
        duration_4=10.0,
        duration_5=10.0,
        duration_6=10.0,
        duration_7=10.0,
        duration_8=10.0,
        duration_9=10.0,
        duration_10=10.0,
        duration_11=10.0,
        duration_12=10.0,
        duration_13=10.0,
        duration_14=10.0,
        duration_15=10.0,
        duration_16=10.0,
        duration_17=10.0,
        duration_18=10.0,
        duration_19=10.0,
        duration_20=10.0,
        duration_21=10.0,
        duration_22=10.0,
        duration_23=10.0,
        duration_24=10.0,
        duration_25=10.0,
        duration_26=10.0,
        duration_27=10.0,
        duration_28=10.0,
        duration_29=10.0,
        duration_30=10.0,
        duration_31=10.0,
        duration_32=10.0,
        prompt_1="",
        prompt_2="",
        prompt_3="",
        prompt_4="",
        prompt_5="",
        prompt_6="",
        prompt_7="",
        prompt_8="",
        prompt_9="",
        prompt_10="",
        prompt_11="",
        prompt_12="",
        prompt_13="",
        prompt_14="",
        prompt_15="",
        prompt_16="",
        prompt_17="",
        prompt_18="",
        prompt_19="",
        prompt_20="",
        prompt_21="",
        prompt_22="",
        prompt_23="",
        prompt_24="",
        prompt_25="",
        prompt_26="",
        prompt_27="",
        prompt_28="",
        prompt_29="",
        prompt_30="",
        prompt_31="",
        prompt_32="",
        segment_index=0,
        render_id="",
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        **kwargs,
    ):
        requested_image_count = min(max(int(image_count), 1), MAX_STORYBOARD_SLOTS)
        durations = [
            float(duration_1), float(duration_2), float(duration_3), float(duration_4), float(duration_5), float(duration_6),
            float(duration_7), float(duration_8), float(duration_9), float(duration_10), float(duration_11), float(duration_12),
            float(duration_13), float(duration_14), float(duration_15), float(duration_16), float(duration_17), float(duration_18),
            float(duration_19), float(duration_20), float(duration_21), float(duration_22), float(duration_23), float(duration_24),
            float(duration_25), float(duration_26), float(duration_27), float(duration_28), float(duration_29), float(duration_30),
            float(duration_31), float(duration_32),
        ]
        prompts = [
            str(prompt_1), str(prompt_2), str(prompt_3), str(prompt_4), str(prompt_5), str(prompt_6), str(prompt_7), str(prompt_8),
            str(prompt_9), str(prompt_10), str(prompt_11), str(prompt_12), str(prompt_13), str(prompt_14), str(prompt_15), str(prompt_16),
            str(prompt_17), str(prompt_18), str(prompt_19), str(prompt_20), str(prompt_21), str(prompt_22), str(prompt_23), str(prompt_24),
            str(prompt_25), str(prompt_26), str(prompt_27), str(prompt_28), str(prompt_29), str(prompt_30), str(prompt_31), str(prompt_32),
        ]
        image_values = {
            1: image_1,
            2: image_2,
            3: image_3,
            4: image_4,
            5: image_5,
            6: image_6,
            7: image_7,
            8: image_8,
            9: image_9,
            10: image_10,
        }
        image_values.update(_extract_indexed_kwargs(kwargs, "image_"))

        connected_image_indexes = sorted(index for index, value in image_values.items() if value is not None)
        image_count = requested_image_count
        csv_durations = _parse_duration_list(durations_csv)

        scheduled_images = []
        last_image = image_1
        for index in range(1, image_count + 1):
            image_value = image_values.get(index)
            if image_value is not None:
                last_image = image_value
            scheduled_images.append(last_image)

        scheduled_durations = []
        for index in range(1, image_count + 1):
            if csv_durations:
                duration_value = csv_durations[min(index - 1, len(csv_durations) - 1)]
            else:
                duration_value = durations[min(index - 1, len(durations) - 1)]
            scheduled_durations.append(max(0.1, float(duration_value)))

        scheduled_prompts = []
        last_prompt = prompts[0] if prompts else ""
        for index in range(1, image_count + 1):
            prompt_value = prompts[min(index - 1, len(prompts) - 1)] if prompts else ""
            if str(prompt_value).strip():
                last_prompt = str(prompt_value)
            scheduled_prompts.append(last_prompt if str(last_prompt).strip() else str(prompt_value))

        active_segment_index = int(segment_index) if str(render_id or "").strip() else 0
        selected_audio, selected_image, current_segment, total_segments, segment_duration_seconds, debug_text = _select_storyboard_segment(
            audio,
            scheduled_images,
            scheduled_durations,
            start_time=start_time,
            end_time=end_time,
            segment_index=active_segment_index,
        )
        selected_prompt = scheduled_prompts[min(current_segment, len(scheduled_prompts) - 1)] if scheduled_prompts else ""
        debug_text = f"{debug_text} | dynamic_images={connected_image_indexes} | prompt_chars={len(str(selected_prompt).strip())} | render_id={render_id or 'fresh'}"
        logging.info("[LTX Motion] Storyboard prompt segment selector %s", debug_text)
        return (
            selected_audio,
            selected_image,
            current_segment,
            total_segments,
            float(segment_duration_seconds),
            selected_prompt,
            debug_text,
        )


class LTXMotionWaveformStoryboardSelector:
    @classmethod
    def INPUT_TYPES(cls):
        audio_files = _list_input_audio_files()
        if not audio_files:
            audio_files = [""]

        required = {
            "audio": ("AUDIO",),
            "image_1": ("IMAGE",),
            "audio_file": (audio_files, {"default": audio_files[0]}),
            "image_count": ("INT", {"default": 1, "min": 1, "max": MAX_STORYBOARD_SLOTS, "step": 1}),
            "keyframes_json": ("STRING", {"default": "[0.0]", "multiline": False}),
        }
        optional = {
            "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            "render_id": ("STRING", {"default": ""}),
        }
        for index in range(2, MAX_STORYBOARD_SLOTS + 1):
            optional[f"image_{index}"] = ("IMAGE",)

        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "image", "current_segment", "total_segments", "segment_duration_seconds", "debug_text")
    FUNCTION = "select_segment"
    CATEGORY = "LTX Motion/audio"

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file="", image_count=1, keyframes_json="[0.0]", render_id="", input_types=None, **kwargs):
        try:
            if audio_file:
                _resolve_audio_path(audio_file)
            normalized_keyframes = _normalize_keyframe_list(_parse_keyframe_list(keyframes_json))
            if len(normalized_keyframes) > MAX_STORYBOARD_SLOTS:
                return f"Too many keyframes: {len(normalized_keyframes)} exceeds {MAX_STORYBOARD_SLOTS}"
            logging.info(
                "[LTX Motion][validate] Waveform storyboard selector: audio_file=%s image_count=%s keyframes=%s input_types=%s",
                audio_file,
                image_count,
                normalized_keyframes,
                input_types,
            )
        except Exception as exc:
            return str(exc)
        return True

    def select_segment(
        self,
        audio,
        image_1,
        audio_file="",
        image_count=1,
        keyframes_json="[0.0]",
        segment_index=0,
        render_id="",
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        **kwargs,
    ):
        image_values = {
            1: image_1,
            2: image_2,
            3: image_3,
            4: image_4,
            5: image_5,
            6: image_6,
            7: image_7,
            8: image_8,
            9: image_9,
            10: image_10,
        }
        image_values.update(_extract_indexed_kwargs(kwargs, "image_"))

        waveform, sample_rate = _normalize_audio_tensor(audio)
        total_duration = waveform.shape[-1] / sample_rate if sample_rate else 0.0
        keyframes = _normalize_keyframe_list(_parse_keyframe_list(keyframes_json), total_duration=total_duration)
        requested_image_count = min(max(int(image_count), 1), MAX_STORYBOARD_SLOTS)
        image_count = max(requested_image_count, len(keyframes))
        image_count = min(image_count, MAX_STORYBOARD_SLOTS)
        keyframes = keyframes[:image_count]

        scheduled_images = []
        last_image = image_1
        connected_image_indexes = sorted(index for index, value in image_values.items() if value is not None)
        for index in range(1, image_count + 1):
            image_value = image_values.get(index)
            if image_value is not None:
                last_image = image_value
            scheduled_images.append(last_image)

        active_segment_index = int(segment_index) if str(render_id or "").strip() else 0
        selected_audio, selected_image, current_segment, total_segments, segment_duration_seconds, debug_text = _select_keyframed_storyboard_segment(
            audio,
            scheduled_images,
            keyframes,
            segment_index=active_segment_index,
            max_segment_seconds=30.0,
        )
        debug_text = f"{debug_text} | dynamic_images={connected_image_indexes} | audio_file={audio_file} | render_id={render_id or 'fresh'}"
        logging.info("[LTX Motion] Waveform storyboard selector %s", debug_text)
        return (
            selected_audio,
            selected_image,
            current_segment,
            total_segments,
            float(segment_duration_seconds),
            debug_text,
        )


class LTXMotionStoryboardPromptSelector:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "current_segment": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            "prompt_count": ("INT", {"default": 1, "min": 1, "max": MAX_STORYBOARD_SLOTS, "step": 1}),
        }
        optional = {
            "segment_count": ("INT", {"default": 0, "forceInput": True}),
        }
        for index in range(1, MAX_STORYBOARD_SLOTS + 1):
            required[f"prompt_{index}"] = ("STRING", {"default": "", "multiline": True})

        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("selected_prompt", "debug_text")
    FUNCTION = "select_prompt"
    CATEGORY = "LTX Motion/audio"

    @classmethod
    def VALIDATE_INPUTS(cls, current_segment=0, prompt_count=1, segment_count=0, **_kwargs):
        try:
            int(current_segment)
            count_source = int(segment_count) if int(segment_count or 0) > 0 else int(prompt_count)
            min(max(count_source, 1), MAX_STORYBOARD_SLOTS)
        except Exception as exc:
            return str(exc)
        return True

    def select_prompt(
        self,
        current_segment=0,
        prompt_count=1,
        segment_count=0,
        prompt_1="",
        prompt_2="",
        prompt_3="",
        prompt_4="",
        prompt_5="",
        prompt_6="",
        prompt_7="",
        prompt_8="",
        prompt_9="",
        prompt_10="",
        **kwargs,
    ):
        prompt_values = {
            1: prompt_1,
            2: prompt_2,
            3: prompt_3,
            4: prompt_4,
            5: prompt_5,
            6: prompt_6,
            7: prompt_7,
            8: prompt_8,
            9: prompt_9,
            10: prompt_10,
        }
        prompt_values.update(_extract_indexed_kwargs(kwargs, "prompt_"))

        resolved_segment_count = int(segment_count or 0)
        active_prompt_count = resolved_segment_count if resolved_segment_count > 0 else int(prompt_count)
        active_prompt_count = min(max(active_prompt_count, 1), MAX_STORYBOARD_SLOTS)
        selected_index = max(0, min(int(current_segment), active_prompt_count - 1))

        normalized_prompts = []
        normalized_sources = []
        filled_prompt_indexes = []
        last_prompt = ""
        last_source_index = 1
        for index in range(1, active_prompt_count + 1):
            raw_prompt = prompt_values.get(index)
            prompt_text = "" if raw_prompt is None else str(raw_prompt)
            if prompt_text.strip():
                last_prompt = prompt_text
                last_source_index = index
                filled_prompt_indexes.append(index)

            normalized_prompts.append(last_prompt if last_prompt else prompt_text)
            normalized_sources.append(last_source_index if last_prompt else index)

        selected_prompt = normalized_prompts[selected_index] if normalized_prompts else ""
        selected_source_index = normalized_sources[selected_index] if normalized_sources else 1
        count_source = "segment_count" if resolved_segment_count > 0 else "prompt_count"
        debug_text = (
            f"current_segment={selected_index + 1}/{active_prompt_count} | "
            f"count_source={count_source} | "
            f"source_prompt=prompt_{selected_source_index} | "
            f"filled_prompts={filled_prompt_indexes} | chars={len(selected_prompt.strip())}"
        )
        logging.info("[LTX Motion] Storyboard prompt selector %s", debug_text)
        return (selected_prompt, debug_text)


class LTXMotionStoryboardFileListSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "image_paths": ("STRING", {"default": "ref2.png\nimage_2.png\nimage_3.png", "multiline": True}),
                "durations_csv": ("STRING", {"default": "5\n10\n10", "multiline": True}),
                "start_time": ("STRING", {"default": "", "multiline": False}),
                "end_time": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "image", "current_segment", "total_segments", "segment_duration_seconds", "debug_text")
    FUNCTION = "select_segment"
    CATEGORY = "LTX Motion/audio"

    @classmethod
    def VALIDATE_INPUTS(cls, image_paths="", durations_csv="", start_time="", end_time="", **kwargs):
        try:
            paths = _parse_line_list(image_paths)
            if not paths:
                return "image_paths must contain at least one image path"
            for image_path in paths:
                _resolve_image_path(image_path)

            durations = _parse_duration_list(durations_csv)
            if not durations:
                return "durations_csv must contain at least one duration"

            start_seconds = _parse_optional_timecode(start_time, 0.0)
            end_text = "" if end_time is None else str(end_time).strip()
            end_seconds = None if not end_text else _parse_timecode(end_text)
            if end_seconds is not None and end_seconds <= start_seconds:
                return "end_time must be greater than start_time"
        except Exception as exc:
            return str(exc)
        return True

    def select_segment(self, audio, image_paths, durations_csv, start_time="", end_time="", segment_index=0):
        paths = _parse_line_list(image_paths)
        durations = _parse_duration_list(durations_csv)
        if not paths:
            raise ValueError("image_paths must contain at least one image path")
        if not durations:
            raise ValueError("durations_csv must contain at least one duration")

        selected_audio, selected_path, current_segment, total_segments, segment_duration_seconds, debug_text = _select_storyboard_segment(
            audio,
            paths,
            durations,
            start_time=start_time,
            end_time=end_time,
            segment_index=segment_index,
        )
        selected_image = _load_image_tensor(selected_path)
        debug_text = f"{debug_text} | path={selected_path}"
        logging.info("[LTX Motion] Storyboard file selector %s", debug_text)
        return (
            selected_audio,
            selected_image,
            current_segment,
            total_segments,
            float(segment_duration_seconds),
            debug_text,
        )

class LTXMotionAudioSegmentLoop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "video/LTX_2.3_ia2v"}),
                "merge_segments": ("BOOLEAN", {"default": True}),
                "keep_segments": ("BOOLEAN", {"default": True}),
                "render_id": ("STRING", {"default": ""}),
                "segment_base_name": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "queue_next"
    OUTPUT_NODE = True
    CATEGORY = "LTX Motion/workflow"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        video=None,
        current_segment=None,
        total_segments=None,
        enabled=True,
        filename_prefix="video/LTX_2.3_ia2v",
        merge_segments=True,
        keep_segments=True,
        render_id="",
        segment_base_name="",
        prompt=None,
        unique_id=None,
        extra_pnginfo=None,
        input_types=None,
        **kwargs,
    ):
        try:
            video_info = None
            if video is not None:
                if hasattr(video, "get_dimensions"):
                    video_info = {"dimensions": video.get_dimensions()}
                else:
                    video_info = type(video).__name__
            logging.info(
                "[LTX Motion][validate] Loop inputs: current_segment=%s total_segments=%s enabled=%s prefix=%s merge=%s keep=%s render_id=%s segment_base_name=%s video=%s prompt_present=%s unique_id=%s extra_pnginfo_present=%s input_types=%s kwargs=%s",
                current_segment,
                total_segments,
                enabled,
                filename_prefix,
                merge_segments,
                keep_segments,
                render_id,
                segment_base_name,
                video_info,
                prompt is not None,
                unique_id,
                extra_pnginfo is not None,
                input_types,
                sorted(kwargs.keys()),
            )
            _log_prompt_cycle_excerpt("[LTX Motion][validate] Loop", prompt)
        except Exception as exc:
            logging.exception("[LTX Motion][validate] Failed to log loop inputs: %s", exc)
        return True

    def queue_next(
        self,
        video,
        current_segment,
        total_segments,
        enabled=True,
        filename_prefix="video/LTX_2.3_ia2v",
        merge_segments=True,
        keep_segments=True,
        render_id="",
        segment_base_name="",
        prompt=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        current_segment = int(current_segment)
        total_segments = int(total_segments)
        enabled = bool(enabled)
        merge_segments = bool(merge_segments)
        keep_segments = bool(keep_segments)
        filename_prefix = filename_prefix or "video/LTX_2.3_ia2v"
        render_id = render_id or ""
        segment_base_name = segment_base_name or ""

        if not render_id and not segment_base_name and current_segment > 0:
            raise RuntimeError(
                f"Fresh run is starting from segment {current_segment + 1} instead of segment 1. "
                "Reset the storyboard selector segment_index to 0 or reload the workflow before running again."
            )

        active_render_id = render_id or uuid.uuid4().hex[:10]

        _log_prompt_cycle_excerpt("[LTX Motion][execute] Loop", prompt)

        output_folder, resolved_base_name, _subfolder = _resolve_output_location(filename_prefix, video)
        active_base_name = segment_base_name or resolved_base_name
        segment_name = _segment_filename(active_base_name, active_render_id, current_segment)
        segment_path = os.path.join(output_folder, segment_name)

        effective_prompt = prompt
        effective_extra_pnginfo = extra_pnginfo
        try:
            live_prompt, live_extra_data, _, _ = _get_current_queue_item()
            if live_prompt is not None:
                effective_prompt = live_prompt
            if effective_extra_pnginfo is None:
                effective_extra_pnginfo = live_extra_data.get("extra_pnginfo", None)
        except Exception:
            pass

        metadata = _build_metadata(effective_prompt, effective_extra_pnginfo)
        frame_summary = _summarize_video_frames(video)
        logging.info("[LTX Motion] Segment %s/%s frame summary before save: %s", current_segment + 1, total_segments, frame_summary)
        video = _sanitize_video_images(video)
        video = _sanitize_video_audio(video)
        video.save_to(segment_path, format=Types.VideoContainer("auto"), codec="auto", metadata=metadata)
        logging.info("[LTX Motion] Saved segment %s/%s to %s", current_segment + 1, total_segments, segment_path)

        if not enabled:
            return {"ui": {"text": [f"Saved segment {current_segment + 1}/{total_segments}: {segment_name}"]}}

        next_segment = current_segment + 1
        if next_segment >= total_segments:
            ui_text = [f"Saved segment {current_segment + 1}/{total_segments}: {segment_name}"]

            if merge_segments:
                segment_paths = [
                    os.path.join(output_folder, _segment_filename(active_base_name, active_render_id, index))
                    for index in range(total_segments)
                ]
                final_name = _final_filename(active_base_name, active_render_id)
                final_path = os.path.join(output_folder, final_name)
                _concat_segments(segment_paths, final_path)
                logging.info("[LTX Motion] Merged %s segments into %s", total_segments, final_path)
                ui_text.append(f"Merged final video: {final_name}")

                if not keep_segments:
                    for path in segment_paths:
                        if os.path.exists(path):
                            os.remove(path)

            return {"ui": {"text": ui_text}}

        base_prompt = effective_prompt
        if base_prompt is None:
            live_prompt, _, _, _ = _get_current_queue_item()
            base_prompt = live_prompt
        prompt_copy = copy.deepcopy(_normalize_prompt_keys(base_prompt))
        loader_updated = False
        loop_updated = False
        for node_id, node in prompt_copy.items():
            if node.get("class_type") in {"LTXMotionSegmentedAudioLoader", "LTXMotionStoryboardSegmentSelector", "LTXMotionWaveformStoryboardSelector"}:
                node.setdefault("inputs", {})["segment_index"] = next_segment
                node.setdefault("inputs", {})["render_id"] = active_render_id
                loader_updated = True
            is_current_loop = unique_id is not None and node_id == str(unique_id)
            is_loop_fallback = unique_id is None and node.get("class_type") == "LTXMotionAudioSegmentLoop"
            if is_current_loop or is_loop_fallback:
                node.setdefault("inputs", {})["render_id"] = active_render_id
                node.setdefault("inputs", {})["segment_base_name"] = active_base_name
                loop_updated = True

        if not loader_updated:
            raise ValueError("LTXMotionAudioSegmentLoop could not find a compatible segment loader node in the prompt.")
        if not loop_updated:
            raise ValueError("LTXMotionAudioSegmentLoop could not update its render_id in the prompt.")

        logging.info("[LTX Motion] Requeueing segment %s/%s", next_segment + 1, total_segments)
        _enqueue_prompt(prompt_copy)
        return {"ui": {"text": [f"Saved segment {current_segment + 1}/{total_segments}: {segment_name}", f"Queued segment {next_segment + 1}/{total_segments}"]}}


class LTXMotionSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "video/LTX_2.3_ia2v_range"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "LTX Motion/workflow"

    def save_video(self, video, filename_prefix="video/LTX_2.3_ia2v_range", prompt=None, extra_pnginfo=None):
        output_path, output_name = _save_video_output(video, filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)
        logging.info("[LTX Motion] Saved final video to %s", output_path)
        return {"ui": {"text": [f"Saved final video: {output_name}"]}}


NODE_CLASS_MAPPINGS = {
    "LTXMotionSegmentedAudioLoader": LTXMotionSegmentedAudioLoader,
    "LTXMotionAudioRangeExtractor": LTXMotionAudioRangeExtractor,
    "LTXMotionStoryboardSegmentSelector": LTXMotionStoryboardSegmentSelector,
    "LTXMotionStoryboardSegmentPromptSelector": LTXMotionStoryboardSegmentPromptSelector,
    "LTXMotionWaveformStoryboardSelector": LTXMotionWaveformStoryboardSelector,
    "LTXMotionStoryboardPromptSelector": LTXMotionStoryboardPromptSelector,
    "LTXMotionStoryboardFileListSelector": LTXMotionStoryboardFileListSelector,
    "LTXMotionDebugImageStats": LTXMotionDebugImageStats,
    "LTXMotionDebugLatentStats": LTXMotionDebugLatentStats,
    "LTXMotionAudioSegmentLoop": LTXMotionAudioSegmentLoop,
    "LTXMotionSaveVideo": LTXMotionSaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXMotionSegmentedAudioLoader": "LTX Motion Segmented Audio Loader",
    "LTXMotionAudioRangeExtractor": "LTX Motion Audio Range Extractor",
    "LTXMotionStoryboardSegmentSelector": "LTX Motion Storyboard Segment Selector",
    "LTXMotionStoryboardSegmentPromptSelector": "LTX Motion Storyboard Segment Prompt Selector",
    "LTXMotionWaveformStoryboardSelector": "LTX Motion Waveform Storyboard Selector",
    "LTXMotionStoryboardPromptSelector": "LTX Motion Storyboard Prompt Selector",
    "LTXMotionStoryboardFileListSelector": "LTX Motion Storyboard File List Selector",
    "LTXMotionDebugImageStats": "LTX Motion Debug Image Stats",
    "LTXMotionDebugLatentStats": "LTX Motion Debug Latent Stats",
    "LTXMotionAudioSegmentLoop": "LTX Motion Audio Segment Loop",
    "LTXMotionSaveVideo": "LTX Motion Save Video",
}