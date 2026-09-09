// =====================================================
// LOCAL WHISPER - Browser Speech-to-Text
// =====================================================

import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";


// =====================================================
// Configuration
// =====================================================

// Allow models to be downloaded from Hugging Face
env.allowLocalModels = false;

// Cache downloaded models in the browser
env.useBrowserCache = true;


// =====================================================
// HTML elements
// =====================================================

const audioFile =
    document.getElementById("audioFile");

const fileInfo =
    document.getElementById("fileInfo");

const fileName =
    document.getElementById("fileName");

const fileSize =
    document.getElementById("fileSize");

const removeFile =
    document.getElementById("removeFile");

const transcribeButton =
    document.getElementById("transcribeButton");

const dropZone =
    document.getElementById("dropZone");

const progressArea =
    document.getElementById("progressArea");

const progressBar =
    document.getElementById("progressBar");

const status =
    document.getElementById("status");

const resultSection =
    document.getElementById("resultSection");

const transcription =
    document.getElementById("transcription");

const resultStatus =
    document.getElementById("resultStatus");

const downloadTxt =
    document.getElementById("downloadTxt");

const downloadSrt =
    document.getElementById("downloadSrt");

const language =
    document.getElementById("language");

const modelSelect =
    document.getElementById("model");


// =====================================================
// Variables
// =====================================================

let selectedFile = null;

let transcriber = null;

let currentModel = null;


// =====================================================
// File selection
// =====================================================

audioFile.addEventListener(
    "change",
    function (event) {

        const file =
            event.target.files[0];

        if (!file) {
            return;
        }

        handleFile(file);

    }
);


// =====================================================
// Handle file
// =====================================================

function handleFile(file) {

    selectedFile = file;

    console.log(
        "Selected:",
        file.name,
        file.type,
        file.size
    );


    fileName.textContent =
        file.name;

    fileSize.textContent =
        formatFileSize(file.size);


    fileInfo.classList.remove(
        "hidden"
    );


    transcribeButton.disabled =
        false;


    dropZone.classList.add(
        "file-selected"
    );

}


// =====================================================
// Format file size
// =====================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {

        return bytes + " B";

    }

    if (bytes < 1024 * 1024) {

        return (
            (bytes / 1024).toFixed(1)
            + " KB"
        );

    }

    return (
        (bytes / (1024 * 1024)).toFixed(1)
        + " MB"
    );

}


// =====================================================
// Drag and drop
// =====================================================

dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.classList.add(
            "dragging"
        );

    }
);


dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.classList.remove(
            "dragging"
        );

    }
);


dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.classList.remove(
            "dragging"
        );

        const file =
            event.dataTransfer.files[0];

        if (file) {

            handleFile(file);

        }

    }
);


// =====================================================
// Remove file
// =====================================================

removeFile.addEventListener(
    "click",
    function () {

        selectedFile = null;

        audioFile.value = "";

        fileInfo.classList.add(
            "hidden"
        );

        transcribeButton.disabled =
            true;

        resultSection.classList.add(
            "hidden"
        );

    }
);


// =====================================================
// Load Whisper
// =====================================================

async function loadWhisper() {

    const selectedModel =
        modelSelect.value;


    // Don't reload the same model

    if (
        transcriber &&
        currentModel === selectedModel
    ) {

        return;

    }


    progressArea.classList.remove(
        "hidden"
    );


    status.textContent =
        "Loading Whisper model...";


    progressBar.style.width =
        "10%";


    console.log(
        "Loading model:",
        selectedModel
    );


    try {

        transcriber =
            await pipeline(
                "automatic-speech-recognition",

                `onnx-community/whisper-${selectedModel}`,

                {
                    device:
                        "webgpu",

                    dtype:
                        "q4"
                }
            );


        currentModel =
            selectedModel;


        progressBar.style.width =
            "100%";


        status.textContent =
            "Whisper model ready.";

    }


    catch (webgpuError) {

        console.warn(
            "WebGPU failed:",
            webgpuError
        );


        status.textContent =
            "WebGPU unavailable. Using CPU...";


        progressBar.style.width =
            "30%";


        try {

            transcriber =
                await pipeline(
                    "automatic-speech-recognition",

                    `onnx-community/whisper-${selectedModel}`,

                    {
                        dtype:
                            "q4"
                    }
                );


            currentModel =
                selectedModel;


            progressBar.style.width =
                "100%";


            status.textContent =
                "Whisper model ready.";

        }


        catch (cpuError) {

            console.error(
                "Whisper failed:",
                cpuError
            );


            throw cpuError;

        }

    }

}


// =====================================================
// Decode audio
// =====================================================

async function decodeAudio(file) {

    const arrayBuffer =
        await file.arrayBuffer();


    const audioContext =
        new AudioContext({
            sampleRate: 16000
        });


    const audioBuffer =
        await audioContext.decodeAudioData(
            arrayBuffer
        );


    const channelData =
        audioBuffer.getChannelData(0);


    await audioContext.close();


    return channelData;

}


// =====================================================
// Transcribe
// =====================================================

transcribeButton.addEventListener(
    "click",
    async function () {

        if (!selectedFile) {

            alert(
                "Please select an audio file first."
            );

            return;

        }


        try {

            transcribeButton.disabled =
                true;


            resultSection.classList.add(
                "hidden"
            );


            progressArea.classList.remove(
                "hidden"
            );


            status.textContent =
                "Starting Whisper...";


            progressBar.style.width =
                "5%";


            // -----------------------------------------
            // Load model
            // -----------------------------------------

            await loadWhisper();


            // -----------------------------------------
            // Decode audio
            // -----------------------------------------

            status.textContent =
                "Reading audio...";


            progressBar.style.width =
                "35%";


            const audio =
                await decodeAudio(
                    selectedFile
                );


            console.log(
                "Audio samples:",
                audio.length
            );


            // -----------------------------------------
            // Options
            // -----------------------------------------

            const options = {

                return_timestamps:
                    true,

                chunk_length_s:
                    30,

                stride_length_s:
                    5

            };


            const selectedLanguage =
                language.value;


            if (
                selectedLanguage !== "auto"
            ) {

                options.language =
                    selectedLanguage;

            }


            // -----------------------------------------
            // Run Whisper
            // -----------------------------------------

            status.textContent =
                "Transcribing...";


            progressBar.style.width =
                "50%";


            const result =
                await transcriber(
                    audio,
                    options
                );


            console.log(
                "Whisper result:",
                result
            );


            // -----------------------------------------
            // Display result
            // -----------------------------------------

            transcription.value =
                result.text || "";


            resultStatus.textContent =
                "Transcription completed successfully.";


            progressBar.style.width =
                "100%";


            status.textContent =
                "Done!";


            resultSection.classList.remove(
                "hidden"
            );


            // Save timestamp chunks

            window.whisperChunks =
                result.chunks || [];


        }


        catch (error) {

            console.error(
                "Transcription error:",
                error
            );


            status.textContent =
                "Transcription failed.";


            progressBar.style.width =
                "0%";


            alert(
                "Transcription failed.\n\n" +
                error.message
            );

        }


        finally {

            transcribeButton.disabled =
                false;

        }

    }
);


// =====================================================
// Download TXT
// =====================================================

downloadTxt.addEventListener(
    "click",
    function () {

        const text =
            transcription.value;


        downloadFile(
            text,
            createFilename(
                ".txt"
            ),
            "text/plain"
        );

    }
);


// =====================================================
// Download SRT
// =====================================================

downloadSrt.addEventListener(
    "click",
    function () {

        const chunks =
            window.whisperChunks || [];


        let srt = "";


        chunks.forEach(
            (chunk, index) => {

                if (
                    !chunk.timestamp ||
                    chunk.timestamp.length < 2
                ) {

                    return;

                }


                const start =
                    chunk.timestamp[0];


                const end =
                    chunk.timestamp[1];


                if (
                    start == null ||
                    end == null
                ) {

                    return;

                }


                srt +=
                    (index + 1) +
                    "\n";


                srt +=
                    formatSrtTime(start) +
                    " --> " +
                    formatSrtTime(end) +
                    "\n";


                srt +=
                    (chunk.text || "").trim() +
                    "\n\n";

            }
        );


        if (!srt.trim()) {

            srt =
                "1\n" +
                "00:00:00,000 --> 00:00:10,000\n" +
                transcription.value.trim() +
                "\n";

        }


        downloadFile(
            srt,
            createFilename(
                ".srt"
            ),
            "text/plain"
        );

    }
);


// =====================================================
// SRT timestamp
// =====================================================

function formatSrtTime(seconds) {

    seconds =
        Math.max(
            0,
            seconds
        );


    const hours =
        Math.floor(
            seconds / 3600
        );


    const minutes =
        Math.floor(
            (seconds % 3600) / 60
        );


    const secs =
        Math.floor(
            seconds % 60
        );


    const milliseconds =
        Math.floor(
            (seconds % 1) * 1000
        );


    return (

        String(hours).padStart(2, "0")

        + ":" +

        String(minutes).padStart(2, "0")

        + ":" +

        String(secs).padStart(2, "0")

        + "," +

        String(milliseconds).padStart(3, "0")

    );

}


// =====================================================
// Download helper
// =====================================================

function downloadFile(
    content,
    filename,
    type
) {

    const blob =
        new Blob(
            [content],
            {
                type:
                    type
            }
        );


    const url =
        URL.createObjectURL(
            blob
        );


    const link =
        document.createElement(
            "a"
        );


    link.href =
        url;


    link.download =
        filename;


    document.body.appendChild(
        link
    );


    link.click();


    link.remove();


    URL.revokeObjectURL(
        url
    );

}


// =====================================================
// Filename
// =====================================================

function createFilename(extension) {

    if (!selectedFile) {

        return (
            "transcription" +
            extension
        );

    }


    const original =
        selectedFile.name;


    const dot =
        original.lastIndexOf(".");


    const base =
        dot > 0
            ? original.substring(
                0,
                dot
            )
            : original;


    return (
        base +
        extension
    );

}


// =====================================================
// Startup
// =====================================================

console.log(
    "Local Whisper loaded successfully."
);
