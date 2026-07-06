import { app } from "../../scripts/app.js";

const TARGET_CLASS = "LTXMotionStoryboardSegmentSelector";
const IMAGE_PREFIX = "image_";
const DURATION_PREFIX = "duration_";
const MIN_IMAGE_COUNT = 1;
const MAX_IMAGE_COUNT = 32;
const MIN_NODE_WIDTH = 420;
const audioDurationCache = new Map();

function getImageIndex(name) {
    return Number.parseInt(String(name).slice(IMAGE_PREFIX.length), 10);
}

function getImageInputs(node) {
    return (node.inputs || [])
        .filter((input) => input?.name?.startsWith(IMAGE_PREFIX))
        .sort((left, right) => getImageIndex(left.name) - getImageIndex(right.name));
}

function getImageCountWidget(node) {
    return (node.widgets || []).find((widget) => widget?.name === "image_count");
}

function getDurationWidgets(node) {
    return (node.widgets || [])
        .filter((widget) => widget?.name?.startsWith(DURATION_PREFIX))
        .sort((left, right) => getImageIndex(left.name.replace(DURATION_PREFIX, IMAGE_PREFIX)) - getImageIndex(right.name.replace(DURATION_PREFIX, IMAGE_PREFIX)));
}

function getWidget(node, name) {
    return (node.widgets || []).find((widget) => widget?.name === name);
}

function clampImageCount(value) {
    const numeric = Number.parseInt(value, 10);
    if (!Number.isFinite(numeric)) {
        return MIN_IMAGE_COUNT;
    }
    return Math.min(MAX_IMAGE_COUNT, Math.max(MIN_IMAGE_COUNT, numeric));
}

function parseOptionalTimecode(value) {
    const text = String(value ?? "").trim();
    if (!text) {
        return null;
    }

    if (/^\d+(\.\d+)?$/.test(text)) {
        return Number.parseFloat(text);
    }

    const parts = text.split(":").map((part) => Number.parseFloat(part.trim()));
    if (parts.some((part) => !Number.isFinite(part))) {
        return null;
    }
    if (parts.length === 2) {
        return (parts[0] * 60) + parts[1];
    }
    if (parts.length === 3) {
        return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
    }
    return null;
}

function formatSeconds(seconds) {
    const clamped = Math.max(0, Number(seconds) || 0);
    const hours = Math.floor(clamped / 3600);
    const minutes = Math.floor((clamped % 3600) / 60);
    const secs = clamped % 60;
    if (hours > 0) {
        return `${hours}:${String(minutes).padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
    }
    return `${minutes}:${secs.toFixed(1).padStart(4, "0")}`;
}

function getGraphLink(graph, linkId) {
    if (!graph || linkId == null) {
        return null;
    }
    if (graph.links && typeof graph.links === "object") {
        return graph.links[linkId] ?? null;
    }
    return null;
}

function getNodeById(graph, nodeId) {
    if (!graph || nodeId == null) {
        return null;
    }
    if (typeof graph.getNodeById === "function") {
        return graph.getNodeById(nodeId);
    }
    return graph._nodes_by_id?.[nodeId] ?? null;
}

function getConnectedAudioFilename(node) {
    const audioInput = (node.inputs || []).find((input) => input?.name === "audio");
    const link = getGraphLink(app.graph, audioInput?.link);
    const originNode = getNodeById(app.graph, link?.origin_id);
    if (!originNode || originNode.type !== "LoadAudio") {
        return null;
    }
    const widgetValues = Array.isArray(originNode.widgets_values) ? originNode.widgets_values : [];
    const audioFile = typeof widgetValues[0] === "string" ? widgetValues[0].trim() : "";
    return audioFile || null;
}

async function getAudioDurationSeconds(audioFile) {
    if (!audioFile) {
        return null;
    }
    if (audioDurationCache.has(audioFile)) {
        return audioDurationCache.get(audioFile);
    }

    const response = await fetch(`/ltx23-motion/audio-waveform?audio_file=${encodeURIComponent(audioFile)}&bins=64`);
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload?.error || "Failed to load audio duration");
    }
    const duration = Number(payload?.duration);
    if (!Number.isFinite(duration)) {
        throw new Error("Audio duration is unavailable");
    }
    audioDurationCache.set(audioFile, duration);
    return duration;
}

function resizeNode(node) {
    const computedSize = node.computeSize?.();
    if (!computedSize) {
        return;
    }

    const currentWidth = Array.isArray(node.size) ? node.size[0] : computedSize[0];
    const imageCount = clampImageCount(getImageCountWidget(node)?.value ?? MIN_IMAGE_COUNT);
    const minimumHeight = 210 + (imageCount * 24);
    node.size = [Math.max(MIN_NODE_WIDTH, currentWidth, computedSize[0]), Math.max(minimumHeight, computedSize[1])];
    app.graph.setDirtyCanvas(true, true);
}

function getHideTargets(widget) {
    const element = widget?.inputEl || widget?.element || widget?.el;
    if (!element) {
        return [];
    }

    const targets = [element];
    if (element.parentElement) {
        targets.push(element.parentElement);
    }
    if (element.parentElement?.parentElement) {
        targets.push(element.parentElement.parentElement);
    }
    return targets.filter((target, index, array) => target && array.indexOf(target) === index);
}

function ensureCounterWidget(node) {
    if (node.__ltxCounterWidget) {
        return node.__ltxCounterWidget;
    }
    if (!Array.isArray(node.widgets)) {
        return null;
    }

    const widget = {
        name: "storyboard_counter",
        type: "storyboard_counter",
        value: "Song counter: connect LoadAudio to show remaining time.",
        options: { serialize: false },
        computeSize(width) {
            return [Math.max(width ?? 320, 320), 30];
        },
        draw(ctx, _node, widgetWidth, y, widgetHeight) {
            const width = Math.max(0, widgetWidth - 20);
            const x = 10;
            const textY = y + (widgetHeight / 2);
            ctx.save();
            ctx.font = "12px sans-serif";
            ctx.fillStyle = widget.__ltxColor || "#d8dee9";
            ctx.textBaseline = "middle";
            const text = String(widget.value || "");
            const clipped = text.length > 96 ? `${text.slice(0, 93)}...` : text;
            ctx.fillText(clipped, x, textY, width);
            ctx.restore();
        },
    };
    node.widgets.push(widget);
    node.__ltxCounterWidgetIndex = node.widgets.length - 1;
    node.__ltxCounterWidget = { widget };
    return node.__ltxCounterWidget;
}

async function updateCounterWidget(node) {
    const counter = ensureCounterWidget(node);
    if (!counter) {
        return;
    }

    const updateToken = (node.__ltxCounterToken || 0) + 1;
    node.__ltxCounterToken = updateToken;

    const imageCount = clampImageCount(getImageCountWidget(node)?.value ?? MIN_IMAGE_COUNT);
    const visibleDurations = getDurationWidgets(node)
        .filter((widget) => Number.parseInt(widget.name.slice(DURATION_PREFIX.length), 10) <= imageCount)
        .map((widget) => Number.parseFloat(widget.value))
        .filter((value) => Number.isFinite(value) && value > 0);
    const scheduledSeconds = visibleDurations.reduce((sum, value) => sum + value, 0);

    const audioFile = getConnectedAudioFilename(node);
    if (!audioFile) {
        counter.widget.value = `Storyboard schedule: ${formatSeconds(scheduledSeconds)} planned. Connect LoadAudio directly to show remaining time.`;
        counter.widget.__ltxColor = "#d8dee9";
        resizeNode(node);
        return;
    }

    try {
        const totalDuration = await getAudioDurationSeconds(audioFile);
        if (node.__ltxCounterToken !== updateToken) {
            return;
        }

        const startSeconds = parseOptionalTimecode(getWidget(node, "start_time")?.value) ?? 0;
        const endSeconds = parseOptionalTimecode(getWidget(node, "end_time")?.value);
        const windowStart = Math.max(0, Math.min(startSeconds, totalDuration));
        const windowEnd = endSeconds == null ? totalDuration : Math.max(windowStart, Math.min(endSeconds, totalDuration));
        const usableWindow = Math.max(0, windowEnd - windowStart);
        const remainingSeconds = usableWindow - scheduledSeconds;
        const leftLabel = remainingSeconds >= 0 ? `left ${formatSeconds(remainingSeconds)}` : `over by ${formatSeconds(Math.abs(remainingSeconds))}`;

        counter.widget.value = `Song window ${formatSeconds(usableWindow)} | scheduled ${formatSeconds(scheduledSeconds)} | ${leftLabel}`;
        counter.widget.__ltxColor = remainingSeconds >= 0 ? "#d8dee9" : "#ff8f8f";
        resizeNode(node);
    } catch (error) {
        if (node.__ltxCounterToken !== updateToken) {
            return;
        }
        counter.widget.value = `Storyboard schedule: ${formatSeconds(scheduledSeconds)} planned. Duration lookup failed for ${audioFile}.`;
        counter.widget.__ltxColor = "#ffd58a";
        resizeNode(node);
    }
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
    if (!widget || widget.__ltxHidden) {
        return;
    }

    widget.__ltxOriginalType = widget.type;
    widget.__ltxOriginalComputeSize = widget.computeSize;
    widget.__ltxOriginalSerializeValue = widget.serializeValue;

    const hideTargets = getHideTargets(widget);
    if (hideTargets.length) {
        widget.__ltxHideTargets = hideTargets.map((target) => ({
            target,
            cssText: target.style.cssText,
        }));
        for (const { target } of widget.__ltxHideTargets) {
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
    widget.__ltxHidden = true;
}

function showWidget(widget) {
    if (!widget || !widget.__ltxHidden) {
        return;
    }

    widget.type = widget.__ltxOriginalType;
    widget.computeSize = widget.__ltxOriginalComputeSize;
    widget.serializeValue = widget.__ltxOriginalSerializeValue;

    if (Array.isArray(widget.__ltxHideTargets)) {
        for (const { target, cssText } of widget.__ltxHideTargets) {
            target.style.cssText = cssText;
        }
    }

    widget.__ltxHidden = false;
}

function syncDurationWidgets(node, activeImageCount) {
    const durationWidgets = getDurationWidgets(node);
    for (const widget of durationWidgets) {
        const index = Number.parseInt(widget.name.slice(DURATION_PREFIX.length), 10);
        if (index <= activeImageCount) {
            showWidget(widget);
        } else {
            hideWidget(widget);
        }
    }

    hideWidget(getWidget(node, "durations_csv"));
    hideWidget(getWidget(node, "render_id"));
}

function syncImageInputs(node) {
    const widget = getImageCountWidget(node);
    const imageCount = clampImageCount(widget?.value ?? MIN_IMAGE_COUNT);

    if (widget) {
        widget.value = imageCount;
    }

    ensureImageInputs(node, imageCount);
    syncDurationWidgets(node, imageCount);
    updateCounterWidget(node);
}

app.registerExtension({
    name: "LTX23Motion.StoryboardDynamicInputs",
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== TARGET_CLASS) {
            return;
        }

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            syncImageInputs(this);
            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = originalOnConnectionsChange?.apply(this, arguments);
            syncImageInputs(this);
            return result;
        };
    },
    async nodeCreated(node) {
        if (node.comfyClass !== TARGET_CLASS) {
            return;
        }

        ensureCounterWidget(node);

        const hookWidget = (widget, onChange) => {
            if (!widget || widget.__ltxMotionDynamicHooked) {
                return;
            }
            const originalCallback = widget.callback;
            widget.callback = function (value) {
                const result = originalCallback?.apply(this, arguments);
                onChange();
                return result;
            };
            widget.__ltxMotionDynamicHooked = true;
        };

        hookWidget(getImageCountWidget(node), () => syncImageInputs(node));
        hookWidget(getWidget(node, "start_time"), () => updateCounterWidget(node));
        hookWidget(getWidget(node, "end_time"), () => updateCounterWidget(node));
        for (const widget of getDurationWidgets(node)) {
            hookWidget(widget, () => updateCounterWidget(node));
        }

        syncImageInputs(node);
    },
});