var dropArea = document.getElementById("drop-area");
var fileListDiv = document.getElementById("file-list");

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
