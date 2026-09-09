import {
    pipeline
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";


// =====================================================
// ELEMENTS
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


// =====================================================
// VARIABLES
// =====================================================

let selectedFile = null;

let transcriber = null;


// =====================================================
// FILE SELECTION
// =====================================================

audioFile.addEventListener(
    "change",
    (event) => {

        const file =
            event.target.files[0];

        if (!file) {
            return;
        }

        selectedFile =
            file;


        fileName.textContent =
            file.name;


        fileSize.textContent =
            formatFileSize(
                file.size
            );


        fileInfo.classList.remove(
            "hidden"
        );


        transcribeButton.disabled =
            false;


        console.log(
            "Audio selected:",
            file.name
        );

    }
);


// =====================================================
// REMOVE FILE
// =====================================================

removeFile.addEventListener(
    "click",
    () => {

        selectedFile =
            null;

        audioFile.value =
            "";

        fileInfo.classList.add(
            "hidden"
        );

        transcribeButton.disabled =
            true;

    }
);


// =====================================================
// FILE SIZE
// =====================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {

        return bytes + " B";

    }

    if (bytes < 1024 * 1024) {

        return (
            (bytes / 1024)
                .toFixed(1)
            + " KB"
        );

    }

    return (
        (bytes / 1024 / 1024)
            .toFixed(1)
        + " MB"
    );

}


// =====================================================
// LOAD WHISPER
// =====================================================

async function createTranscriber() {

    status.textContent =
        "Loading Whisper...";

    progressBar.style.width =
        "10%";


    console.log(
        "Loading Whisper model..."
    );


    // Try WebGPU first

    if (
        navigator.gpu
    ) {

        try {

            console.log(
                "Trying WebGPU..."
            );


            const pipe =
                await pipeline(
                    "automatic-speech-recognition",
                    "onnx-community/whisper-tiny.en",
                    {
                        device:
                            "webgpu"
                    }
                );


            console.log(
                "Whisper loaded with WebGPU."
            );


            return pipe;

        }

        catch (error) {

            console.warn(
                "WebGPU failed.",
                error
            );

        }

    }


    // CPU fallback

    console.log(
        "Using CPU/WASM..."
    );


    status.textContent =
        "WebGPU unavailable. Loading CPU version...";


    const pipe =
        await pipeline(
            "automatic-speech-recognition",
            "onnx-community/whisper-tiny.en"
        );


    console.log(
        "Whisper loaded with CPU."
    );


    return pipe;

}


// =====================================================
// TRANSCRIBE
// =====================================================

transcribeButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {

            alert(
                "Please select an audio file."
            );

            return;

        }


        try {

            transcribeButton.disabled =
                true;


            progressArea.classList.remove(
                "hidden"
            );


            resultSection.classList.add(
                "hidden"
            );


            status.textContent =
                "Starting...";


            progressBar.style.width =
                "5%";


            // -----------------------------------------
            // Load Whisper
            // -----------------------------------------

            if (!transcriber) {

                transcriber =
                    await createTranscriber();

            }


            progressBar.style.width =
                "50%";


            status.textContent =
                "Transcribing audio...";


            // -----------------------------------------
            // Create temporary URL
            // -----------------------------------------

            const audioURL =
                URL.createObjectURL(
                    selectedFile
                );


            console.log(
                "Sending audio to Whisper..."
            );


            // -----------------------------------------
            // Run Whisper
            // -----------------------------------------

            const result =
                await transcriber(
                    audioURL,
                    {
                        return_timestamps:
                            true
                    }
                );


            URL.revokeObjectURL(
                audioURL
            );


            console.log(
                "Whisper result:",
                result
            );


            // -----------------------------------------
            // Show transcript
            // -----------------------------------------

            transcription.value =
                result.text || "";


            resultStatus.textContent =
                "Transcription completed.";


            progressBar.style.width =
                "100%";


            status.textContent =
                "Done!";


            resultSection.classList.remove(
                "hidden"
            );


            // Store chunks for later

            window.whisperChunks =
                result.chunks || [];

        }


        catch (error) {

            console.error(
                "TRANSCRIPTION ERROR:",
                error
            );


            status.textContent =
                "Error";


            progressBar.style.width =
                "0%";


            alert(
                "Whisper could not transcribe the file.\n\n" +
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
// DOWNLOAD TEXT
// =====================================================

downloadTxt.addEventListener(
    "click",
    () => {

        const text =
            transcription.value;


        const blob =
            new Blob(
                [text],
                {
                    type:
                        "text/plain"
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
            "transcription.txt";


        document.body.appendChild(
            link
        );


        link.click();


        link.remove();


        URL.revokeObjectURL(
            url
        );

    }
);


console.log(
    "Local Whisper ready."
);
