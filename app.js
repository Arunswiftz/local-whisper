import { pipeline, env } from
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

// --------------------------------------------------
// Configuration
// --------------------------------------------------

env.allowLocalModels = false;
env.useBrowserCache = true;


// --------------------------------------------------
// Elements
// --------------------------------------------------

const audioFile = document.getElementById("audioFile");
const dropZone = document.getElementById("dropZone");

const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");

const removeFile = document.getElementById("removeFile");

const transcribeButton =
    document.getElementById("transcribeButton");

const language =
    document.getElementById("language");

const modelSelect =
    document.getElementById("model");

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


// --------------------------------------------------
// Variables
// --------------------------------------------------

let selectedFile = null;
let transcriber = null;
let currentModel = null;


// --------------------------------------------------
// File handling
// --------------------------------------------------

audioFile.addEventListener("change", event => {

    const file = event.target.files[0];

    if (file) {
        selectFile(file);
    }

});


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


function selectFile(file) {

    selectedFile = file;

    fileName.textContent = file.name;

    fileSize.textContent =
        formatBytes(file.size);

    fileInfo.classList.remove("hidden");

    transcribeButton.disabled = false;

}


function formatBytes(bytes) {

    if (bytes === 0) {
        return "0 Bytes";
    }

    const units = [
        "Bytes",
        "KB",
        "MB",
        "GB"
    ];

    const index =
        Math.floor(Math.log(bytes) / Math.log(1024));

    return (
        parseFloat(
            (bytes / Math.pow(1024, index))
                .toFixed(2)
        )
        + " "
        + units[index]
    );

}


removeFile.addEventListener("click", () => {

    selectedFile = null;

    audioFile.value = "";

    fileInfo.classList.add("hidden");

    transcribeButton.disabled = true;

    resultSection.classList.add("hidden");

});


// --------------------------------------------------
// Load Whisper
// --------------------------------------------------

async function loadModel() {

    const selectedModel =
        modelSelect.value;

    if (
        transcriber &&
        currentModel === selectedModel
    ) {
        return;
    }

    progressArea.classList.remove("hidden");

    status.textContent =
        "Loading Whisper model...";

    progressBar.style.width = "10%";

    try {

        transcriber = await pipeline(
            "automatic-speech-recognition",
            `onnx-community/whisper-${selectedModel}`,
            {
                device:
                    "webgpu"
            }
        );

        currentModel = selectedModel;

        progressBar.style.width = "100%";

        status.textContent =
            "Whisper is ready.";

    }

    catch (error) {

        console.error(error);

        status.textContent =
            "WebGPU unavailable. Trying CPU...";

        progressBar.style.width = "20%";

        transcriber = await pipeline(
            "automatic-speech-recognition",
            `onnx-community/whisper-${selectedModel}`
        );

        currentModel = selectedModel;

        progressBar.style.width = "100%";

        status.textContent =
            "Whisper is ready.";

    }

}


// --------------------------------------------------
// Transcription
// --------------------------------------------------

transcribeButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {
            return;
        }

        transcribeButton.disabled = true;

        progressArea.classList.remove("hidden");

        resultSection.classList.add("hidden");

        status.textContent =
            "Preparing audio...";

        progressBar.style.width = "5%";

        try {

            await loadModel();

            status.textContent =
                "Transcribing audio...";

            progressBar.style.width = "40%";

            const audioURL =
                URL.createObjectURL(
                    selectedFile
                );

            const options = {
                return_timestamps: true
            };

            const selectedLanguage =
                language.value;

            if (
                selectedLanguage !== "auto"
            ) {

                options.language =
                    selectedLanguage;

            }

            const result =
                await transcriber(
                    audioURL,
                    options
                );

            URL.revokeObjectURL(audioURL);

            progressBar.style.width =
                "100%";

            status.textContent =
                "Transcription complete.";

            transcription.value =
                result.text || "";

            resultStatus.textContent =
                "Finished successfully.";

            resultSection.classList.remove(
                "hidden"
            );

        }

        catch (error) {

            console.error(error);

            status.textContent =
                "Something went wrong.";

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


// --------------------------------------------------
// Download TXT
// --------------------------------------------------

downloadTxt.addEventListener(
    "click",
    () => {

        const text =
            transcription.value;

        downloadFile(
            text,
            "transcription.txt",
            "text/plain"
        );

    }
);


// --------------------------------------------------
// Download SRT
// --------------------------------------------------

downloadSrt.addEventListener(
    "click",
    () => {

        const text =
            transcription.value;

        const srt =
            createBasicSrt(text);

        downloadFile(
            srt,
            "transcription.srt",
            "text/plain"
        );

    }
);


function downloadFile(
    content,
    filename,
    type
) {

    const blob =
        new Blob(
            [content],
            { type }
        );

    const url =
        URL.createObjectURL(blob);

    const link =
        document.createElement("a");

    link.href = url;

    link.download = filename;

    link.click();

    URL.revokeObjectURL(url);

}


// --------------------------------------------------
// Basic SRT
// --------------------------------------------------

function createBasicSrt(text) {

    return (
        "1\n" +
        "00:00:00,000 --> 00:00:00,000\n" +
        text.trim() +
        "\n"
    );

}
:::

**Important:** this first version's SRT download is intentionally basic. In the next iteration we can use Whisper's actual timestamp chunks to create proper subtitles.
