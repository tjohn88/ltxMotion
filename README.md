# LTX 2.3 Motion Workflows

This folder contains a ComfyUI custom node pack plus workflow variants for LTX 2.3 image-to-video generation, storyboard scheduling, waveform-based segment editing, and per-image prompt rotation.

## Key files

- `LTX-2.3_Image_To_Video_Motion_Transfer.json`
- `install_ltx23_motion_models.bat`
- `merge_safetensors_shards.py`
- `ltx_motion_audio_segments.py`
- `workflows/geekatplay_studio_ltx_2_3_ia2v_storyboard_song_looper_reimport.json`
- `workflows/geekatplay_studio_ltx_2_3_ia2v_storyboard_song_looper_per_image_prompts_reimport.json`
- `workflows/geekatplay_studio_ltx_2_3_ia2v_waveform_storyboard_song_looper_reimport.json`
- `workflows/geekatplay_studio_ltx_2_3_ia2v_waveform_storyboard_song_looper_per_image_prompts_reimport.json`

The `workflows` folder now contains only the workflows shipped with this pack for release.

## Workflow purpose

The included workflows cover:

- motion-transfer generation from an input image and motion reference video
- storyboard-driven audio segmentation with rotating images
- waveform-based keyframe editing for music timing
- per-image prompt selection through `LTXMotionStoryboardPromptSelector`

All workflow variants use the same LTX 2.3 checkpoint, LoRA, and Gemma text encoder destinations described below.

## Required model destinations

- `models/checkpoints/ltx-2.3-22b-dev.safetensors`
- `models/loras/ltx-2.3-22b-distilled-lora-384.safetensors`
- `models/loras/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors`
- `models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors`
- `models/text_encoders/comfy_gemma_3_12B_it.safetensors`
- `models/text_encoders/gemma_3_12B_it.safetensors`

## Installer notes

Run `install_ltx23_motion_models.bat`.

The batch file will:

1. ask where to keep downloaded files
2. ask for the ComfyUI `models` directory
3. ask for an existing Gemma shard folder, defaulting to `models/text_encoders/gemma`
4. download the public LTX 2.3 model files directly into the expected folders
5. download the Comfy-compatible `gemma_3_12B_it_fp4_mixed.safetensors` text encoder when possible
6. if that is not available, merge Gemma immediately if shard files are already present in that folder
7. otherwise optionally download the gated Google Gemma 3 12B shard files if you provide a Hugging Face token with access
8. verify that every workflow-required file exists in the correct destination folder before finishing

The installer also creates compatible aliases for Gemma so both official LTX example names and older workflow names resolve to a tokenizer-capable file.

## Runtime requirements

This repo does not need a separate `pip install -r requirements.txt` step for its own custom nodes when used inside a normal ComfyUI installation.

For the storyboard and waveform song-looper workflows, make sure `ffmpeg` is available. The loop node tries these sources in order:

- `VHS_FORCE_FFMPEG_PATH`
- `imageio_ffmpeg` if it is installed in the active Python environment
- `ffmpeg` on `PATH`

If no ffmpeg executable is available, final segment concat will fail.

## Recommended workflows

- `LTX-2.3_Image_To_Video_Motion_Transfer.json` for the original motion-transfer setup
- `workflows/geekatplay_studio_ltx_2_3_ia2v_storyboard_song_looper_reimport.json` for storyboard-driven image rotation over audio
- `workflows/geekatplay_studio_ltx_2_3_ia2v_storyboard_song_looper_per_image_prompts_reimport.json` for storyboard-driven image rotation plus per-image prompts
- `workflows/geekatplay_studio_ltx_2_3_ia2v_waveform_storyboard_song_looper_reimport.json` for waveform-edited storyboard timing
- `workflows/geekatplay_studio_ltx_2_3_ia2v_waveform_storyboard_song_looper_per_image_prompts_reimport.json` for waveform-edited timing plus per-image prompts

## Gemma gating

The Google Gemma repository is gated. Before running the installer for Gemma, make sure your Hugging Face account has accepted access for:

- `google/gemma-3-12b-it`

If you skip the token prompt, the workflow will still be copied, but it will not run until `models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` is present.

If you already downloaded the shard files manually, the installer can merge them directly from a folder such as `I:\models\text_encoders\gemma`. It accepts either `model.safetensors.index.json` or `gemma-3-12B-it.json` as the shard index file.

The raw merged Google shard output does not embed the tokenizer by default, so the workflow uses the Comfy-compatible Gemma file name instead.

## Frame length note

The workflow output length is controlled by the `number of frames` primitive in `LTX-2.3_Image_To_Video_Motion_Transfer.json`.

The motion guide video should be no longer than the requested output length after LTX frame snapping (`8n + 1`). If the guide video is longer, the local `ComfyUI-LTXVideo` node is patched to crop the extra guide frames automatically instead of failing with `Conditioning frames exceed the length of the latent sequence.`

## Storyboard selector note

`LTX Motion Storyboard Segment Selector` now supports dynamically created `image_n` inputs on the ComfyUI canvas.

- The node starts with one storyboard image input by default.
- Set `image_count` to the exact number of storyboard images you want.
- The node expands or shrinks to exactly that many image inputs.
- The node also shows exactly that many duration widgets, one per storyboard image.
- The backend supports up to 32 storyboard slots in the current implementation.
- `durations_csv` remains available as a compatibility fallback, but the intended UI is the per-image duration widgets shown under the image inputs.

`LTX Motion Storyboard File List Selector` is the cleaner option for real workflows.

- It uses one multiline `image_paths` widget instead of many image input sockets.
- It uses one multiline `durations_csv` widget to schedule those images.
- The updated storyboard song looper workflow uses this cleaner selector so the scheduler node itself no longer needs a wall of image connections.

`LTX Motion Waveform Storyboard Selector` is the visual timeline option for music-driven workflows.

- It keeps the same segment outputs as the storyboard selector so it works with the existing loop node.
- It adds a waveform editor in the node UI with play, scrub, add keyframe, and remove keyframe controls.
- Each waveform keyframe creates one `image_n` input and one storyboard segment.
- Segment duration is computed automatically from the next keyframe and capped at 30 seconds.
- The included workflow `geekatplay_studio_ltx_2_3_ia2v_waveform_storyboard_song_looper_reimport.json` is based on the working storyboard song looper.
- The included workflow `geekatplay_studio_ltx_2_3_ia2v_waveform_storyboard_song_looper_per_image_prompts_reimport.json` adds the separate prompt selector on top of the waveform storyboard path.
- The current waveform renderer is intended for WAV files from the ComfyUI input directory.

`LTX Motion Storyboard Prompt Selector` is the recommended per-image prompt workflow companion.

- It stays as a separate node from the storyboard scheduler.
- It takes `current_segment` plus a synced `segment_count` link from the storyboard scheduler.
- The node UI auto-syncs its visible prompt count from the connected storyboard scheduler so `prompt_1` pairs with `image_1`, `prompt_2` pairs with `image_2`, and so on.
- Blank prompt slots fall back to the most recent non-empty prompt so you can reuse text across consecutive images.
- The workflow `geekatplay_studio_ltx_2_3_ia2v_storyboard_song_looper_per_image_prompts_reimport.json` uses this separate prompt selector and feeds its `selected_prompt` output into `TextGenerateLTX2Prompt`.
- The workflow `geekatplay_studio_ltx_2_3_ia2v_waveform_storyboard_song_looper_per_image_prompts_reimport.json` uses the same selector pattern with the waveform storyboard node.

`LTX Motion Storyboard Segment Prompt Selector` remains available as a combined scheduler-plus-prompts node for future development.