var dropArea = document.getElementById("drop-area");
var fileListDiv = document.getElementById("file-list");
var selectedVideo = document.getElementById("selected-video");

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
  var selectedFile = fileInput.files[0];
    var url = new URL(window.location.href);
    url.searchParams.set('selectedFile', JSON.stringify(selectedFile));
    history.pushState(null, null, url.href);
});
// Function to display a video
function displayVideo(file) {
    // Display the first video in the selected-video element
    var reader = new FileReader();
    reader.onload = function (e) {
        selectedVideo.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Update the handleFiles function to include video handling
function handleFiles(files) {
    fileListDiv.innerHTML = ""; // Clear the file list

    for (var i = 0; i < files.length; i++) {
        var file = files[i];

        if (file.type.startsWith("video/")) {
            // If it's a video, display it
            displayVideo(file);
        } else {
            // File type not allowed
            alert("Only video files are allowed.");
        }

        // Display file information
        var fileInfo = document.createElement("p");
        fileInfo.textContent =
            "ファイル名: " +
            file.name +
            ", サイズ: " +
            file.size +
            " bytes, タイプ: " +
            file.type;
        fileListDiv.appendChild(fileInfo);
    }
}