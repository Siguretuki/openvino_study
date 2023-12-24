var dropArea = document.getElementById("drop-area");
var fileListDiv = document.getElementById("file-list");
var selectedImage = document.getElementById("selected-image");

// ドラッグエンター時の処理
dropArea.addEventListener("dragenter", function (e) {
  e.preventDefault();
  dropArea.style.border = "2px dashed #000";
});

// ドラッグオーバー時の処理
dropArea.addEventListener("dragover", function (e) {
  e.preventDefault();
});

// ドラッグリーブ時の処理
dropArea.addEventListener("dragleave", function () {
  dropArea.style.border = "2px dashed #ccc";
});

// ドロップ時の処理
dropArea.addEventListener("drop", function (e) {
  e.preventDefault();
  dropArea.style.border = "2px dashed #ccc";

  var files = e.dataTransfer.files;
  handleFiles(files);
});

// input要素に変更があった時の処理
var fileInput = document.getElementById("file-input");
fileInput.addEventListener("change", function () {
  var files = fileInput.files;
  handleFiles(files);
});
// ファイルを処理する関数
function handleFiles(files) {
    fileListDiv.innerHTML = ""; // ファイルリストをクリア

    for (var i = 0; i < files.length; i++) {
        var file = files[i];

        // Check if the file is an image or video
        if (file.type.startsWith("image/") || file.type.startsWith("video/")) {
            var fileInfo = document.createElement("p");
            fileInfo.textContent =
                "ファイル名: " +
                file.name +
                ", サイズ: " +
                file.size +
                " bytes, タイプ: " +
                file.type;
            fileListDiv.appendChild(fileInfo);

            // If it's an image, display it
            if (file.type.startsWith("image/")) {
                displayImage(file);
            }
        } else {
            // File type not allowed
            alert("Only image and video files are allowed.");
        }
    }
}

// Function to display an image
function displayImage(file) {
    // Display the first image in the selected-image element
    var reader = new FileReader();
    reader.onload = function (e) {
        selectedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
}
