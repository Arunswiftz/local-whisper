import {
    pipeline,
    AutoProcessor,
    AutoModelForAudioFrameClassification
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";


// =====================================================
// LOCAL WHISPER
// Browser-only transcription + word timestamps +
// automatic speaker diarization.
// =====================================================


// -----------------------------------------------------
// HTML ELEMENTS
// -----------------------------------------------------

const audioFile = document.getElementById("audioFile");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFile = document.getElementById("removeFile");
const transcribeButton = document.getElementById("transcribeButton");
const dropZone = document.getElementById("dropZone");
const progressArea = document.getElementById("progressArea");
const progressBar = document.getElementById("progressBar");
const status = document.getElementById("status");
const resultSection = document.getElementById("resultSection");
const transcription = document.getElementById("transcription");
const resultStatus = document.getElementById("resultStatus");
const downloadTxt = document.getElementById("downloadTxt");
const downloadSrt = document.getElementById("downloadSrt");
const languageSelect = document.getElementById("language");
const modelSelect = document.getElementById("model");
const sensitivitySelect = document.getElementById("sensitivity");


// -----------------------------------------------------
// VARIABLES
// -----------------------------------------------------

let selectedFile = null;
let transcriber = null;
let segmentationProcessor = null;
let segmentationModel = null;
let modelsLoaded = false;
let cancelled = false;
let transcriptionData = [];


// -----------------------------------------------------
// MODEL SETTINGS
// -----------------------------------------------------

const ASR_MODEL = "onnx-community/whisper-base_timestamped";
const SEGMENTATION_MODEL = "onnx-community/pyannote-segmentation-3.0";

// Lower no_speech_threshold = more willing to keep quiet speech.
// This is deliberately exposed as a user-facing capture sensitivity.
const SENSITIVITY = {
    high: {
        noSpeechThreshold: 0.15,
        label: "High"
    },
    balanced: {
        noSpeechThreshold: 0.25,
        label: "Balanced"
    },
    strict: {
        noSpeechThreshold: 0.40,
        label: "Strict"
    }
};

const SPEAKER_CONFIDENCE_THRESHOLD = 0.20;
const MAX_SRT_BLOCK_SECONDS = 7;
const SPEAKER_MERGE_GAP_SECONDS = 1.25;


// -----------------------------------------------------
// ENGLISH VARIANT SETTINGS
// -----------------------------------------------------

// Whisper has one English language token rather than separate
// acoustic models for US, Australian, and UK English. The selected
// variant therefore biases spelling, vocabulary, punctuation, and
// conventions through the initial prompt.

const ENGLISH_VARIANTS = {
    "en-US": {
        label: "English US",
        prompt:
            "Transcribe in American English. Use US English spelling, vocabulary, punctuation, and conventions."
    },
    "en-AU": {
        label: "English Australia",
        prompt:
            "Transcribe in Australian English. Use Australian English spelling, vocabulary, punctuation, and conventions."
    },
    "en-GB": {
        label: "English UK",
        prompt:
            "Transcribe in British English. Use UK English spelling, vocabulary, punctuation, and conventions."
    }
};


function getSelectedEnglishVariant() {
    return ENGLISH_VARIANTS[languageSelect.value] || ENGLISH_VARIANTS["en-US"];
}


function getSelectedSensitivity() {
    return SENSITIVITY[sensitivitySelect.value] || SENSITIVITY.balanced;
}


// -----------------------------------------------------
// FILE SELECTION
// -----------------------------------------------------

audioFile.addEventListener("change", event => {
    const file = event.target.files[0];

    if (file) {
        selectFile(file);
    }
});


function selectFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    fileInfo.classList.remove("hidden");
    transcribeButton.disabled = false;
    resultSection.classList.add("hidden");

    console.log("Selected:", file.name);
}


function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + " B";
    }

    if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + " KB";
    }

    return (bytes / 1024 / 1024).toFixed(1) + " MB";
}


// -----------------------------------------------------
// REMOVE FILE
// -----------------------------------------------------

removeFile.addEventListener("click", () => {
    selectedFile = null;
    transcriptionData = [];
    audioFile.value = "";
    fileInfo.classList.add("hidden");
    resultSection.classList.add("hidden");
    transcribeButton.disabled = true;
});


// -----------------------------------------------------
// DRAG AND DROP
// -----------------------------------------------------

dropZone.addEventListener("dragover", event => {
    event.preventDefault();
    dropZone.classList.add("dragging");
});


dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
});


dropZone.addEventListener("drop", event => {
    event.preventDefault();
    dropZone.classList.remove("dragging");

    const file = event.dataTransfer.files[0];

    if (file) {
        selectFile(file);
    }
});


// -----------------------------------------------------
// LOAD WHISPER + SPEAKER SEGMENTATION
// -----------------------------------------------------

async function loadModels() {
    if (modelsLoaded) {
        return;
    }

    status.textContent = "Loading Whisper Base and speaker detection models...";
    progressBar.style.width = "5%";

    let device = "wasm";

    if ("gpu" in navigator) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                device = "webgpu";
            }
        }
        catch (error) {
            console.warn("WebGPU detection failed; using WASM.", error);
        }
    }

    console.log("Loading models on:", device);

    const whisperOptions = device === "webgpu"
        ? {
            device: "webgpu",
            dtype: {
                encoder_model: "fp32",
                decoder_model_merged: "q4"
            }
        }
        : {
            device: "wasm",
            dtype: "q8"
        };

    status.textContent = "Loading Whisper Base...";

    transcriber = await pipeline(
        "automatic-speech-recognition",
        ASR_MODEL,
        whisperOptions
    );

    progressBar.style.width = "45%";

    status.textContent = "Loading speaker detection model...";

    // PyAnnote segmentation currently runs through WASM even when
    // Whisper is accelerated with WebGPU.
    segmentationProcessor = await AutoProcessor.from_pretrained(
        SEGMENTATION_MODEL
    );

    segmentationModel = await AutoModelForAudioFrameClassification.from_pretrained(
        SEGMENTATION_MODEL,
        {
            device: "wasm",
            dtype: "fp32"
        }
    );

    progressBar.style.width = "55%";
    modelsLoaded = true;

    console.log("Whisper and speaker detection models ready.");
}


// -----------------------------------------------------
// DECODE AUDIO
// -----------------------------------------------------

async function decodeAudio(file) {
    status.textContent = "Reading and normalising audio to 16 kHz mono...";
    progressBar.style.width = "58%";

    const buffer = await file.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(buffer);
    const duration = audioBuffer.duration;

    const targetSampleRate = 16000;
    const numberOfSamples = Math.ceil(duration * targetSampleRate);

    const offlineContext = new OfflineAudioContext(
        1,
        numberOfSamples,
        targetSampleRate
    );

    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start();

    const rendered = await offlineContext.startRendering();
    const samples = rendered.getChannelData(0);

    await audioContext.close();

    console.log("Decoded samples:", samples.length);

    return {
        samples,
        duration,
        sampleRate: targetSampleRate
    };
}


// -----------------------------------------------------
// SPEAKER DIARIZATION
// -----------------------------------------------------

async function detectSpeakers(samples) {
    status.textContent = "Detecting speaker changes...";
    progressBar.style.width = "82%";

    const inputs = await segmentationProcessor(samples);
    const output = await segmentationModel(inputs);

    const segments = segmentationProcessor.post_process_speaker_diarization(
        output.logits,
        samples.length
    )[0];

    const labels = segmentationModel.config.id2label || {};

    const usefulSegments = segments
        .map(segment => ({
            ...segment,
            label: labels[segment.id] || segment.label || `SPEAKER_${segment.id}`
        }))
        .filter(segment => {
            const confidence = Number(segment.confidence ?? 1);
            return Number.isFinite(segment.start) &&
                Number.isFinite(segment.end) &&
                segment.end > segment.start &&
                confidence >= SPEAKER_CONFIDENCE_THRESHOLD;
        });

    console.log("Speaker segments:", usefulSegments);

    return mergeSpeakerSegments(usefulSegments);
}


function mergeSpeakerSegments(segments) {
    const merged = [];

    for (const segment of segments) {
        const previous = merged[merged.length - 1];

        if (
            previous &&
            previous.label === segment.label &&
            segment.start - previous.end <= SPEAKER_MERGE_GAP_SECONDS
        ) {
            previous.end = Math.max(previous.end, segment.end);
            previous.confidence = Math.max(
                previous.confidence ?? 0,
                segment.confidence ?? 0
            );
        }
        else {
            merged.push({ ...segment });
        }
    }

    return merged;
}


function getSpeakerForWord(word, speakerSegments, previousSpeaker) {
    const timestamp = word.timestamp || [];
    const start = Number(timestamp[0]);
    const end = Number(timestamp[1]);

    if (!Number.isFinite(start)) {
        return previousSpeaker || "Speaker 1";
    }

    const safeEnd = Number.isFinite(end) ? end : start + 0.05;
    const midpoint = (start + safeEnd) / 2;

    let best = null;
    let bestOverlap = 0;

    for (const segment of speakerSegments) {
        const overlapStart = Math.max(start, segment.start);
        const overlapEnd = Math.min(safeEnd, segment.end);
        const overlap = Math.max(0, overlapEnd - overlapStart);

        if (overlap > bestOverlap || (overlap === 0 && midpoint >= segment.start && midpoint <= segment.end && !best)) {
            best = segment;
            bestOverlap = overlap;
        }
    }

    return best?.label || previousSpeaker || "Speaker 1";
}


// -----------------------------------------------------
// SPEAKER LABEL NORMALISATION
// -----------------------------------------------------

function createSpeakerNameMap(items) {
    const map = new Map();
    let nextNumber = 1;

    for (const item of items) {
        if (!map.has(item.rawSpeaker)) {
            map.set(item.rawSpeaker, `Speaker ${nextNumber}`);
            nextNumber += 1;
        }
    }

    return map;
}


// -----------------------------------------------------
// BUILD SPEAKER-LABELLED TRANSCRIPT
// -----------------------------------------------------

function buildTranscript(wordChunks, speakerSegments) {
    const words = [];
    let previousSpeaker = null;

    for (const chunk of wordChunks || []) {
        const text = (chunk.text || "").trim();
        const timestamp = chunk.timestamp || [];

        if (!text) {
            continue;
        }

        const start = Number(timestamp[0]);
        const end = Number(timestamp[1]);

        if (!Number.isFinite(start)) {
            continue;
        }

        const rawSpeaker = getSpeakerForWord(
            chunk,
            speakerSegments,
            previousSpeaker
        );

        previousSpeaker = rawSpeaker;

        words.push({
            text,
            start,
            end: Number.isFinite(end) ? end : start + 0.1,
            rawSpeaker
        });
    }

    const speakerMap = createSpeakerNameMap(words);

    for (const word of words) {
        word.speaker = speakerMap.get(word.rawSpeaker) || "Speaker 1";
    }

    const blocks = [];

    for (const word of words) {
        const previous = blocks[blocks.length - 1];

        if (
            previous &&
            previous.speaker === word.speaker &&
            word.start - previous.end <= SPEAKER_MERGE_GAP_SECONDS
        ) {
            previous.words.push(word);
            previous.end = Math.max(previous.end, word.end);
        }
        else {
            blocks.push({
                speaker: word.speaker,
                start: word.start,
                end: word.end,
                words: [word]
            });
        }
    }

    return splitLongBlocks(blocks);
}


function splitLongBlocks(blocks) {
    const output = [];

    for (const block of blocks) {
        let current = null;

        for (const word of block.words) {
            if (!current) {
                current = {
                    speaker: block.speaker,
                    start: word.start,
                    end: word.end,
                    words: [word]
                };
                continue;
            }

            const wouldBeTooLong = word.end - current.start > MAX_SRT_BLOCK_SECONDS;

            if (wouldBeTooLong) {
                output.push(current);
                current = {
                    speaker: block.speaker,
                    start: word.start,
                    end: word.end,
                    words: [word]
                };
            }
            else {
                current.words.push(word);
                current.end = word.end;
            }
        }

        if (current) {
            output.push(current);
        }
    }

    return output.map(block => ({
        ...block,
        text: cleanBlockText(block.words.map(word => word.text).join(" "))
    }));
}


function cleanBlockText(text) {
    return text
        .replace(/\s+([,.!?;:])/g, "$1")
        .replace(/\s+(['’])/g, "$1")
        .replace(/\s+/g, " ")
        .trim();
}


// -----------------------------------------------------
// TRANSCRIBE
// -----------------------------------------------------

transcribeButton.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("Please select an audio file.");
        return;
    }

    cancelled = false;
    transcriptionData = [];

    try {
        transcribeButton.disabled = true;
        resultSection.classList.add("hidden");
        progressArea.classList.remove("hidden");
        transcription.value = "";

        const selectedVariant = getSelectedEnglishVariant();
        const sensitivity = getSelectedSensitivity();

        await loadModels();

        const audio = await decodeAudio(selectedFile);

        if (cancelled) {
            throw new Error("Transcription cancelled.");
        }

        status.textContent =
            `Transcribing with ${selectedVariant.label} • ${sensitivity.label} capture...`;
        progressBar.style.width = "60%";

        const asrResult = await transcriber(audio.samples, {
            language: "english",
            task: "transcribe",
            initial_prompt: selectedVariant.prompt,
            return_timestamps: "word",
            chunk_length_s: 30,
            no_speech_threshold: sensitivity.noSpeechThreshold,
            logprob_threshold: -1.0,
            compression_ratio_threshold: 2.4
        });

        if (cancelled) {
            throw new Error("Transcription cancelled.");
        }

        const wordChunks = asrResult.chunks || [];

        console.log("Whisper word chunks:", wordChunks.length);

        const speakerSegments = await detectSpeakers(audio.samples);

        if (cancelled) {
            throw new Error("Transcription cancelled.");
        }

        transcriptionData = buildTranscript(
            wordChunks,
            speakerSegments
        );

        const textOutput = transcriptionData
            .map(block => `${block.speaker}\n${block.text}`)
            .join("\n\n");

        transcription.value = textOutput;

        progressBar.style.width = "100%";
        status.textContent = "Transcription and speaker detection complete!";

        const speakerCount = new Set(
            transcriptionData.map(block => block.speaker)
        ).size;

        resultStatus.textContent =
            `Processed ${formatTime(audio.duration)} of audio • ` +
            `${speakerCount || 1} speaker(s) detected • ` +
            `${selectedVariant.label} • ${sensitivity.label} capture.`;

        resultSection.classList.remove("hidden");

        console.log("Final speaker-labelled transcript:", transcriptionData);
    }
    catch (error) {
        console.error("TRANSCRIPTION ERROR:", error);

        status.textContent = "Transcription failed.";
        progressBar.style.width = "0%";

        alert(
            "Transcription failed:\n\n" +
            (error?.message || error)
        );
    }
    finally {
        transcribeButton.disabled = !selectedFile;
    }
});


// -----------------------------------------------------
// FORMAT TIME
// -----------------------------------------------------

function formatTime(seconds) {
    const safeSeconds = Math.max(0, Number(seconds) || 0);
    const hours = Math.floor(safeSeconds / 3600);
    const minutes = Math.floor((safeSeconds % 3600) / 60);
    const secs = Math.floor(safeSeconds % 60);

    if (hours > 0) {
        return (
            String(hours).padStart(2, "0") + ":" +
            String(minutes).padStart(2, "0") + ":" +
            String(secs).padStart(2, "0")
        );
    }

    return (
        String(minutes).padStart(2, "0") + ":" +
        String(secs).padStart(2, "0")
    );
}


function formatSrtTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor((safe % 3600) / 60);
    const secs = Math.floor(safe % 60);
    const milliseconds = Math.round((safe - Math.floor(safe)) * 1000);

    let ms = milliseconds;
    let s = secs;
    let m = minutes;
    let h = hours;

    if (ms >= 1000) {
        ms = 0;
        s += 1;
    }

    if (s >= 60) {
        s = 0;
        m += 1;
    }

    if (m >= 60) {
        m = 0;
        h += 1;
    }

    return (
        String(h).padStart(2, "0") + ":" +
        String(m).padStart(2, "0") + ":" +
        String(s).padStart(2, "0") + "," +
        String(ms).padStart(3, "0")
    );
}


// -----------------------------------------------------
// DOWNLOAD TXT
// -----------------------------------------------------

downloadTxt.addEventListener("click", () => {
    if (!transcription.value.trim()) {
        return;
    }

    downloadBlob(
        transcription.value,
        getOutputFilename("txt"),
        "text/plain;charset=utf-8"
    );
});


// -----------------------------------------------------
// DOWNLOAD SRT
// -----------------------------------------------------

downloadSrt.addEventListener("click", () => {
    if (!transcriptionData.length) {
        alert("Please transcribe an audio file first.");
        return;
    }

    const srt = transcriptionData
        .map((block, index) => {
            const start = formatSrtTime(block.start);
            const end = formatSrtTime(Math.max(block.end, block.start + 0.1));
            const text = `${block.speaker}\n${block.text}`;

            return `${index + 1}\n${start} --> ${end}\n${text}`;
        })
        .join("\n\n");

    downloadBlob(
        srt + "\n",
        getOutputFilename("srt"),
        "application/x-subrip;charset=utf-8"
    );
});


function downloadBlob(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
}


// -----------------------------------------------------
// OUTPUT FILENAME
// -----------------------------------------------------

function getOutputFilename(extension = "txt") {
    if (!selectedFile) {
        return `transcription.${extension}`;
    }

    const name = selectedFile.name;
    const dot = name.lastIndexOf(".");
    const base = dot > 0 ? name.substring(0, dot) : name;

    return `${base}_transcription.${extension}`;
}


// -----------------------------------------------------
// STARTUP
// -----------------------------------------------------

console.log("Local Whisper speaker diarization version ready.");
