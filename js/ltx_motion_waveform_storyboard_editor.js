import { app } from "../../scripts/app.js";

const TARGET_CLASS = "LTXMotionWaveformStoryboardSelector";
const IMAGE_PREFIX = "image_";
const MIN_IMAGE_COUNT = 1;
const MAX_IMAGE_COUNT = 32;
const DEFAULT_WIDTH = 820;
const DEFAULT_HEIGHT = 430;

function getImageIndex(name) {
    return Number.parseInt(String(name).slice(IMAGE_PREFIX.length), 10);
}

function clampImageCount(value) {
    const numeric = Number.parseInt(value, 10);
    if (!Number.isFinite(numeric)) {
        return MIN_IMAGE_COUNT;
    }
    return Math.min(MAX_IMAGE_COUNT, Math.max(MIN_IMAGE_COUNT, numeric));
}

function getWidget(node, name) {
    return (node.widgets || []).find((widget) => widget?.name === name);
}

function getImageInputs(node) {
    return (node.inputs || [])
        .filter((input) => input?.name?.startsWith(IMAGE_PREFIX))
        .sort((left, right) => getImageIndex(left.name) - getImageIndex(right.name));
}

function resizeNode(node) {
    const computedSize = node.computeSize?.();
    const width = Math.max(DEFAULT_WIDTH, computedSize?.[0] ?? 0, Array.isArray(node.size) ? node.size[0] : 0);
    const height = Math.max(DEFAULT_HEIGHT, computedSize?.[1] ?? 0, Array.isArray(node.size) ? node.size[1] : 0);
    node.size = [width, height];
    app.graph.setDirtyCanvas(true, true);
}

function ensureImageInputs(node, activeImageCount) {
    const desiredVisibleInputs = clampImageCount(activeImageCount);
    let currentInputs = getImageInputs(node);
    let maxIndex = currentInputs.length ? getImageIndex(currentInputs[currentInputs.length - 1].name) : 1;

    while (maxIndex < desiredVisibleInputs) {
        maxIndex += 1;
        node.addInput(`${IMAGE_PREFIX}${maxIndex}`, "IMAGE");
    }

    currentInputs = getImageInputs(node);
    for (let index = currentInputs.length - 1; index >= 0; index -= 1) {
        const input = currentInputs[index];
        const inputIndex = getImageIndex(input.name);
        if (inputIndex <= desiredVisibleInputs) {
            continue;
        }

        const slot = node.inputs.indexOf(input);
        if (slot !== -1) {
            node.removeInput(slot);
        }
    }

    resizeNode(node);
}

function hideWidget(widget) {
    if (!widget || widget.__ltxWaveHidden) {
        return;
    }

    widget.__ltxWaveOriginalType = widget.type;
    widget.__ltxWaveOriginalComputeSize = widget.computeSize;
    widget.__ltxWaveOriginalSerializeValue = widget.serializeValue;

    const element = widget.inputEl || widget.element || widget.el;
    const targets = [element, element?.parentElement, element?.parentElement?.parentElement].filter(Boolean);
    if (targets.length) {
        widget.__ltxWaveElements = targets.map((target) => ({
            target,
            cssText: target.style.cssText,
        }));
        for (const { target } of widget.__ltxWaveElements) {
            target.style.display = "none";
            target.style.visibility = "hidden";
            target.style.height = "0";
            target.style.minHeight = "0";
            target.style.maxHeight = "0";
            target.style.margin = "0";
            target.style.padding = "0";
            target.style.border = "0";
            target.style.overflow = "hidden";
            target.style.pointerEvents = "none";
        }
    }

    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
    widget.__ltxWaveHidden = true;
}

function showWidget(widget) {
    if (!widget || !widget.__ltxWaveHidden) {
        return;
    }

    widget.type = widget.__ltxWaveOriginalType;
    widget.computeSize = widget.__ltxWaveOriginalComputeSize;
    widget.serializeValue = widget.__ltxWaveOriginalSerializeValue;
    if (Array.isArray(widget.__ltxWaveElements)) {
        for (const { target, cssText } of widget.__ltxWaveElements) {
            target.style.cssText = cssText ?? "";
        }
    }
    widget.__ltxWaveHidden = false;
}

function parseKeyframes(value) {
    try {
        const parsed = JSON.parse(String(value || "[]"));
        if (!Array.isArray(parsed)) {
            return [0];
        }
        return parsed
            .map((item) => Number.parseFloat(item))
            .filter((item) => Number.isFinite(item) && item >= 0)
            .sort((left, right) => left - right);
    } catch {
        return [0];
    }
}

function normalizeKeyframes(keyframes, duration) {
    const unique = [];
    const seen = new Set();
    for (const value of keyframes || []) {
        let seconds = Math.max(0, Number(value) || 0);
        if (Number.isFinite(duration) && duration > 0) {
            seconds = Math.min(seconds, Math.max(0, duration - 0.001));
        }
        const bucket = Math.round(seconds * 1000);
        if (seen.has(bucket)) {
            continue;
        }
        seen.add(bucket);
        unique.push(seconds);
    }

    unique.sort((left, right) => left - right);
    if (!unique.length || unique[0] > 0.001) {
        unique.unshift(0);
    }
    return unique.slice(0, MAX_IMAGE_COUNT);
}

function formatTime(seconds) {
    const clamped = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(clamped / 60);
    const remainder = clamped - minutes * 60;
    return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function buildEditor(node) {
    if (node.__ltxWaveformEditor) {
        return node.__ltxWaveformEditor;
    }

    const container = document.createElement("div");
    container.className = "ltx-waveform-storyboard-editor";
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "8px";
    container.style.padding = "8px 0 4px";
    container.style.minWidth = "760px";

    const toolbar = document.createElement("div");
    toolbar.style.display = "flex";
    toolbar.style.flexWrap = "wrap";
    toolbar.style.gap = "8px";
    toolbar.style.alignItems = "center";

    const playButton = document.createElement("button");
    playButton.textContent = "Play";
    const addButton = document.createElement("button");
    addButton.textContent = "Add keyframe";
    const removeButton = document.createElement("button");
    removeButton.textContent = "Remove selected";

    const status = document.createElement("div");
    status.style.fontSize = "12px";
    status.style.opacity = "0.85";

    toolbar.append(playButton, addButton, removeButton, status);

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "180px";
    canvas.style.border = "1px solid rgba(255,255,255,0.18)";
    canvas.style.borderRadius = "8px";
    canvas.style.background = "#111";
    canvas.style.cursor = "pointer";

    const help = document.createElement("div");
    help.style.fontSize = "12px";
    help.style.opacity = "0.75";
    help.textContent = "Click waveform to scrub. Use Play to listen. Add keyframe at the current playhead. Each marker creates one image input and segment.";

    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.style.display = "none";

    container.append(toolbar, canvas, help, audio);

    const domWidget = typeof node.addDOMWidget === "function"
        ? node.addDOMWidget("waveform_editor", "waveform_editor", container, {
            serialize: false,
            hideOnZoom: false,
            getValue: () => "",
            setValue: () => {},
        })
        : null;
    if (domWidget) {
        domWidget.computeSize = () => [DEFAULT_WIDTH - 30, 240];
    }

    const state = {
        node,
        container,
        canvas,
        audio,
        status,
        playButton,
        addButton,
        removeButton,
        peaks: [],
        duration: 0,
        selectedIndex: 0,
        pointerDown: false,
        error: "",
    };

    node.__ltxWaveformEditor = state;

    const render = () => {
        const width = Math.max(760, Math.floor(container.clientWidth || DEFAULT_WIDTH));
        const height = 180;
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        if (!context) {
            return;
        }

        context.clearRect(0, 0, width, height);
        context.fillStyle = "#101317";
        context.fillRect(0, 0, width, height);

        if (state.error) {
            context.fillStyle = "#ff8f8f";
            context.font = "14px sans-serif";
            context.fillText(state.error, 12, 24);
            return;
        }

        const midY = Math.floor(height / 2);
        context.strokeStyle = "rgba(255,255,255,0.12)";
        context.beginPath();
        context.moveTo(0, midY);
        context.lineTo(width, midY);
        context.stroke();

        if (state.peaks.length) {
            const barWidth = width / state.peaks.length;
            context.fillStyle = "#71d0ff";
            for (let index = 0; index < state.peaks.length; index += 1) {
                const peak = Math.max(0, Math.min(1, state.peaks[index] || 0));
                const barHeight = Math.max(1, peak * (height * 0.42));
                const x = index * barWidth;
                context.fillRect(x, midY - barHeight, Math.max(1, barWidth - 1), barHeight * 2);
            }
        }

        const keyframesWidget = getWidget(node, "keyframes_json");
        const keyframes = normalizeKeyframes(parseKeyframes(keyframesWidget?.value), state.duration);
        keyframes.forEach((time, index) => {
            const ratio = state.duration > 0 ? time / state.duration : 0;
            const x = Math.max(0, Math.min(width, ratio * width));
            context.strokeStyle = index === state.selectedIndex ? "#ffd166" : "#ff7f50";
            context.lineWidth = index === state.selectedIndex ? 3 : 2;
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, height);
            context.stroke();
            context.fillStyle = context.strokeStyle;
            context.font = "12px sans-serif";
            context.fillText(String(index + 1), Math.min(width - 14, x + 4), 14 + (index % 2) * 14);
        });

        if (state.duration > 0) {
            const ratio = Math.max(0, Math.min(1, (audio.currentTime || 0) / state.duration));
            const x = ratio * width;
            context.strokeStyle = "#ffffff";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, height);
            context.stroke();
        }

        status.textContent = `time ${formatTime(audio.currentTime || 0)} / ${formatTime(state.duration)} | keyframes ${keyframes.length} | selected ${state.selectedIndex + 1}`;
    };

    const syncKeyframes = (nextKeyframes, preferredSelectedIndex = null) => {
        const keyframesWidget = getWidget(node, "keyframes_json");
        const imageCountWidget = getWidget(node, "image_count");
        const normalized = normalizeKeyframes(nextKeyframes, state.duration);
        const serialized = JSON.stringify(normalized.map((value) => Number(value.toFixed(3))));
        if (keyframesWidget) {
            keyframesWidget.value = serialized;
        }
        if (imageCountWidget) {
            imageCountWidget.value = normalized.length;
        }
        state.selectedIndex = Math.max(0, Math.min(preferredSelectedIndex ?? state.selectedIndex, normalized.length - 1));
        ensureImageInputs(node, normalized.length);
        render();
    };

    const loadWaveform = async () => {
        const audioFileWidget = getWidget(node, "audio_file");
        const audioFile = String(audioFileWidget?.value || "").trim();
        state.error = "";
        if (!audioFile) {
            state.peaks = [];
            state.duration = 0;
            render();
            return;
        }

        try {
            const response = await fetch(`/ltx23-motion/audio-waveform?audio_file=${encodeURIComponent(audioFile)}&bins=1400`);
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload?.error || "Failed to load waveform");
            }

            state.peaks = Array.isArray(payload.peaks) ? payload.peaks : [];
            state.duration = Number(payload.duration) || 0;
            audio.src = payload.audio_url || `/ltx23-motion/audio-file?audio_file=${encodeURIComponent(audioFile)}`;
            audio.load();
            const currentKeyframes = parseKeyframes(getWidget(node, "keyframes_json")?.value);
            syncKeyframes(currentKeyframes, state.selectedIndex);
        } catch (error) {
            state.error = error?.message || String(error);
            state.peaks = [];
            state.duration = 0;
            render();
        }
    };

    const seekToPosition = (event) => {
        if (!(state.duration > 0)) {
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const offsetX = event.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, offsetX / rect.width));
        audio.currentTime = ratio * state.duration;

        const keyframes = normalizeKeyframes(parseKeyframes(getWidget(node, "keyframes_json")?.value), state.duration);
        let nearestIndex = 0;
        let nearestDistance = Number.POSITIVE_INFINITY;
        for (let index = 0; index < keyframes.length; index += 1) {
            const distance = Math.abs(keyframes[index] - audio.currentTime);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestIndex = index;
            }
        }
        if (nearestDistance <= Math.max(0.2, state.duration / 100)) {
            state.selectedIndex = nearestIndex;
        }
        render();
    };

    playButton.addEventListener("click", async () => {
        if (audio.paused) {
            await audio.play();
        } else {
            audio.pause();
        }
    });

    addButton.addEventListener("click", () => {
        const current = audio.currentTime || 0;
        const keyframes = parseKeyframes(getWidget(node, "keyframes_json")?.value);
        keyframes.push(current);
        const normalized = normalizeKeyframes(keyframes, state.duration);
        const bucket = Math.round(current * 1000);
        const preferredIndex = normalized.findIndex((value) => Math.round(value * 1000) === bucket);
        syncKeyframes(normalized, preferredIndex >= 0 ? preferredIndex : normalized.length - 1);
    });

    removeButton.addEventListener("click", () => {
        const keyframes = normalizeKeyframes(parseKeyframes(getWidget(node, "keyframes_json")?.value), state.duration);
        if (keyframes.length <= 1 || state.selectedIndex <= 0) {
            return;
        }
        keyframes.splice(state.selectedIndex, 1);
        syncKeyframes(keyframes, Math.max(0, state.selectedIndex - 1));
    });

    audio.addEventListener("play", () => {
        playButton.textContent = "Pause";
        render();
    });
    audio.addEventListener("pause", () => {
        playButton.textContent = "Play";
        render();
    });
    audio.addEventListener("timeupdate", render);
    audio.addEventListener("loadedmetadata", render);

    canvas.addEventListener("pointerdown", (event) => {
        state.pointerDown = true;
        seekToPosition(event);
    });
    canvas.addEventListener("pointermove", (event) => {
        if (!state.pointerDown) {
            return;
        }
        seekToPosition(event);
    });
    window.addEventListener("pointerup", () => {
        state.pointerDown = false;
    });

    const audioFileWidget = getWidget(node, "audio_file");
    if (audioFileWidget && !audioFileWidget.__ltxWaveHooked) {
        const originalCallback = audioFileWidget.callback;
        audioFileWidget.callback = function (value) {
            const result = originalCallback?.apply(this, arguments);
            loadWaveform();
            return result;
        };
        audioFileWidget.__ltxWaveHooked = true;
    }

    const imageCountWidget = getWidget(node, "image_count");
    const keyframesWidget = getWidget(node, "keyframes_json");
    const renderIdWidget = getWidget(node, "render_id");
    hideWidget(imageCountWidget);
    hideWidget(keyframesWidget);
    hideWidget(renderIdWidget);
    showWidget(audioFileWidget);
    ensureImageInputs(node, clampImageCount(imageCountWidget?.value ?? 1));

    requestAnimationFrame(() => {
        render();
        loadWaveform();
        resizeNode(node);
    });

    return state;
}

function syncFromStoredState(node) {
    const imageCountWidget = getWidget(node, "image_count");
    const keyframesWidget = getWidget(node, "keyframes_json");
    const renderIdWidget = getWidget(node, "render_id");
    hideWidget(imageCountWidget);
    hideWidget(keyframesWidget);
    hideWidget(renderIdWidget);

    const editor = buildEditor(node);
    const keyframes = normalizeKeyframes(parseKeyframes(keyframesWidget?.value), editor.duration);
    if (imageCountWidget) {
        imageCountWidget.value = keyframes.length;
    }
    ensureImageInputs(node, keyframes.length);
    editor.selectedIndex = Math.max(0, Math.min(editor.selectedIndex, keyframes.length - 1));
    if (editor.container.isConnected) {
        editor.status.textContent = `time ${formatTime(editor.audio.currentTime || 0)} / ${formatTime(editor.duration)} | keyframes ${keyframes.length} | selected ${editor.selectedIndex + 1}`;
    }
}

app.registerExtension({
    name: "LTX23Motion.WaveformStoryboardEditor",
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== TARGET_CLASS) {
            return;
        }

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            syncFromStoredState(this);
            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = originalOnConnectionsChange?.apply(this, arguments);
            syncFromStoredState(this);
            return result;
        };
    },
    async nodeCreated(node) {
        if (node.comfyClass !== TARGET_CLASS) {
            return;
        }
        buildEditor(node);
        syncFromStoredState(node);
    },
});