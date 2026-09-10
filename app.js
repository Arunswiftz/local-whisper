import {
    pipeline
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";


// =====================================================
// LOCAL WHISPER
// Long audio browser transcription
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


// -----------------------------------------------------
// VARIABLES
// -----------------------------------------------------

let selectedFile = null;
let transcriber = null;
let loadedModel = null;
let cancelled = false;


// -----------------------------------------------------
// CHUNK SETTINGS
// -----------------------------------------------------

const CHUNK_SECONDS = 30;
const OVERLAP_SECONDS = 2;


// -----------------------------------------------------
// ENGLISH VARIANT SETTINGS
// -----------------------------------------------------

// Whisper recognizes English as one language. It does not
// have separate acoustic models for US, Australian, and UK
// English. The selected variant is therefore passed to the
// decoder as an English language setting plus a spelling /
// vocabulary prompt so the transcription follows the chosen
// regional English convention as closely as Whisper allows.

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
    return (
        ENGLISH_VARIANTS[languageSelect.value] ||
        ENGLISH_VARIANTS["en-US"]
    );
}


// -----------------------------------------------------
// FILE SELECTION
// -----------------------------------------------------

audioFile.addEventListener("change", event => {
    const file = event.target.files[0];

    if (!file) {
        return;
    }

    selectFile(file);
});


function selectFile(file) {
    selectedFile = file;

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    fileInfo.classList.remove("hidden");
    transcribeButton.disabled = false;
    resultSection.classList.add("hidden");

    console.log("Selected:", file.name);
});


function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + " B";
    }

    if (bytes < 1024 * 1024) {
        return ((bytes / 1024).toFixed(1)) + " KB";
    }

    return ((bytes / 1024 / 1024).toFixed(1)) + " MB";
}


// -----------------------------------------------------
// REMOVE FILE
// -----------------------------------------------------

removeFile.addEventListener("click", () => {
    selectedFile = null;
    audioFile.value = "";
    fileInfo.classList.add("hidden");
    transcribeButton.disabled = true;
    resultSection.classList.add("hidden");
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
// LOAD WHISPER
// -----------------------------------------------------

async function loadWhisper() {
    const selectedModel = modelSelect.value;

    if (transcriber && loadedModel === selectedModel) {
        return;
    }

    transcriber = null;
    loadedModel = null;

    status.textContent = "Loading Whisper model...";
    progressBar.style.width = "5%";

    console.log("Loading Whisper model:", selectedModel);

    const modelName =
        selectedModel === "tiny"
            ? "onnx-community/whisper-tiny"
            : "onnx-community/whisper-tiny.en";

    if ("gpu" in navigator) {
        try {
            status.textContent = "Loading Whisper with WebGPU...";

            transcriber = await pipeline(
                "automatic-speech-recognition",
                modelName,
                {
                    device: "webgpu"
                }
            );

            loadedModel = selectedModel;

            console.log("Whisper WebGPU ready.");
            return;
        }
        catch (error) {
            console.warn("WebGPU failed:", error);
            transcriber = null;
        }
    }

    status.textContent = "Loading CPU version of Whisper...";

    transcriber = await pipeline(
        "automatic-speech-recognition",
        modelName
    );

    loadedModel = selectedModel;

    console.log("Whisper CPU ready.");
}


// -----------------------------------------------------
// DECODE AUDIO
// -----------------------------------------------------

async function decodeAudio(file) {
    status.textContent = "Reading audio file...";
    progressBar.style.width = "8%";

    const buffer = await file.arrayBuffer();
    const audioContext = new AudioContext();

    const audioBuffer = await audioContext.decodeAudioData(buffer);
    const duration = audioBuffer.duration;

    console.log("Audio duration:", duration, "seconds");

    const targetSampleRate = 16000;

    const numberOfSamples = Math.ceil(
        duration * targetSampleRate
    );

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
// CREATE AUDIO CHUNK
// -----------------------------------------------------

function getAudioChunk(samples, startSample, endSample) {
    return samples.slice(startSample, endSample);
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

    try {
        transcribeButton.disabled = true;
        resultSection.classList.add("hidden");
        progressArea.classList.remove("hidden");
        transcription.value = "";

        const selectedVariant = getSelectedEnglishVariant();

        console.log(
            "Selected English variant:",
            selectedVariant.label
        );

        await loadWhisper();

        const audio = await decodeAudio(selectedFile);
        const samples = audio.samples;
        const duration = audio.duration;
        const sampleRate = audio.sampleRate;

        const chunkSamples = CHUNK_SECONDS * sampleRate;
        const overlapSamples = OVERLAP_SECONDS * sampleRate;

        const totalChunks = Math.ceil(
            samples.length / chunkSamples
        );

        console.log("Total chunks:", totalChunks);

        let fullText = "";

        for (let i = 0; i < totalChunks; i++) {
            if (cancelled) {
                throw new Error("Transcription cancelled.");
            }

            const normalStart = i * chunkSamples;

            const normalEnd = Math.min(
                normalStart + chunkSamples,
                samples.length
            );

            const start = Math.max(
                0,
                normalStart - (
                    i > 0 ? overlapSamples : 0
                )
            );

            const end = Math.min(
                samples.length,
                normalEnd + (
                    i < totalChunks - 1
                        ? overlapSamples
                        : 0
                )
            );

            const chunk = getAudioChunk(
                samples,
                start,
                end
            );

            const percent = 10 + ((i / totalChunks) * 85);
            progressBar.style.width = percent + "%";

            const chunkStart = normalStart / sampleRate;
            const chunkEnd = normalEnd / sampleRate;

            status.textContent =
                "Transcribing " +
                selectedVariant.label +
                " • Chunk " +
                (i + 1) +
                " of " +
                totalChunks +
                " • " +
                formatTime(chunkStart) +
                " → " +
                formatTime(chunkEnd);

            console.log(
                `Chunk ${i + 1}/${totalChunks}`,
                chunk.length
            );

            const result = await transcriber(chunk, {
                language: "english",
                task: "transcribe",
                initial_prompt: selectedVariant.prompt,
                return_timestamps: false
            });

            const text = (result.text || "").trim();

            console.log("Chunk result:", text);

            if (text) {
                if (fullText) {
                    fullText += " ";
                }

                fullText += text;
                transcription.value = fullText;
            }

            await sleep(50);
        }

        progressBar.style.width = "100%";
        status.textContent = "Transcription complete!";

        resultStatus.textContent =
            "Processed " +
            formatTime(duration) +
            " of audio as " +
            selectedVariant.label +
            ".";

        transcription.value = fullText;
        resultSection.classList.remove("hidden");

        console.log("Final transcription:", fullText);
    }
    catch (error) {
        console.error("TRANSCRIPTION ERROR:", error);

        status.textContent = "Transcription failed.";
        progressBar.style.width = "0%";

        alert(
            "Transcription failed:\n\n" +
            error.message
        );
    }
    finally {
        transcribeButton.disabled = false;
    }
});


// -----------------------------------------------------
// RESET MODEL WHEN MODEL SELECTION CHANGES
// -----------------------------------------------------

modelSelect.addEventListener("change", () => {
    transcriber = null;
    loadedModel = null;
});


// -----------------------------------------------------
// FORMAT TIME
// -----------------------------------------------------

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

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


// -----------------------------------------------------
// SLEEP
// -----------------------------------------------------

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


// -----------------------------------------------------
// DOWNLOAD TXT
// -----------------------------------------------------

downloadTxt.addEventListener("click", () => {
    if (!transcription.value.trim()) {
        return;
    }

    const blob = new Blob(
        [transcription.value],
        {
            type: "text/plain;charset=utf-8"
        }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = getOutputFilename();

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
});


// -----------------------------------------------------
// DOWNLOAD SRT
// -----------------------------------------------------

// SRT timestamp generation will be added when the
// transcription pipeline is upgraded to retain Whisper
// segment timestamps.

downloadSrt.addEventListener("click", () => {
    alert(
        "SRT export will be available once timestamped transcription is enabled."
    );
});


// -----------------------------------------------------
// OUTPUT FILENAME
// -----------------------------------------------------

function getOutputFilename() {
    if (!selectedFile) {
        return "transcription.txt";
    }

    const name = selectedFile.name;
    const dot = name.lastIndexOf(".");

    const base = dot > 0
        ? name.substring(0, dot)
        : name;

    return base + "_transcription.txt";
}


// -----------------------------------------------------
// STARTUP
// -----------------------------------------------------

console.log("Local Whisper English variant version ready.");
