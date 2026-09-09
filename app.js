// =====================================================
// LOCAL WHISPER - Browser Speech to Text
// =====================================================

console.log("Local Whisper starting...");


// -----------------------------------------------------
// Get HTML elements
// -----------------------------------------------------

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


// -----------------------------------------------------
// Check that elements exist
// -----------------------------------------------------

console.log("Audio input:", audioFile);
console.log("File info:", fileInfo);
console.log("Transcribe button:", transcribeButton);


// -----------------------------------------------------
// Selected file
// -----------------------------------------------------

let selectedFile = null;


// -----------------------------------------------------
// File selected using the button
// -----------------------------------------------------

audioFile.addEventListener(
    "change",
    function (event) {

        console.log("File input changed");

        const files =
            event.target.files;

        if (!files || files.length === 0) {

            console.log("No file selected");

            return;
        }

        const file = files[0];

        console.log(
            "Selected file:",
            file.name,
            file.type,
            file.size
        );

        handleFile(file);
    }
);


// -----------------------------------------------------
// Handle selected file
// -----------------------------------------------------

function handleFile(file) {

    selectedFile = file;

    console.log(
        "Handling file:",
        file.name
    );


    // Show filename

    fileName.textContent =
        file.name;


    // Show size

    fileSize.textContent =
        formatFileSize(file.size);


    // Show file information

    fileInfo.classList.remove(
        "hidden"
    );


    // Enable transcription button

    transcribeButton.disabled =
        false;


    // Change drop-zone appearance

    dropZone.classList.add(
        "file-selected"
    );

}


// -----------------------------------------------------
// Format file size
// -----------------------------------------------------

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
        (bytes / (1024 * 1024))
            .toFixed(1)
        + " MB"
    );

}


// -----------------------------------------------------
// Remove selected file
// -----------------------------------------------------

removeFile.addEventListener(
    "click",
    function () {

        console.log(
            "Removing selected file"
        );

        selectedFile = null;

        audioFile.value = "";

        fileInfo.classList.add(
            "hidden"
        );

        transcribeButton.disabled =
            true;

        dropZone.classList.remove(
            "file-selected"
        );

    }
);


// -----------------------------------------------------
// Drag and drop
// -----------------------------------------------------

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

        const files =
            event.dataTransfer.files;

        if (!files || files.length === 0) {
            return;
        }

        const file = files[0];

        console.log(
            "Dropped file:",
            file.name
        );

        handleFile(file);

    }
);


// -----------------------------------------------------
// Transcribe button
// -----------------------------------------------------

transcribeButton.addEventListener(
    "click",
    async function () {

        if (!selectedFile) {

            alert(
                "Please select an audio file first."
            );

            return;
        }


        console.log(
            "Starting transcription:",
            selectedFile.name
        );


        // Temporary message

        alert(
            "File received successfully!\n\n" +
            "File: " +
            selectedFile.name +
            "\n\n" +
            "The upload system is working.\n" +
            "Next we will connect Whisper."
        );

    }
);


// -----------------------------------------------------
// Startup
// -----------------------------------------------------

console.log(
    "Local Whisper is ready."
);
