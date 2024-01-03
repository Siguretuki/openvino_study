// file-upload.js

document.addEventListener("DOMContentLoaded", function () {
  var dropArea = document.getElementById("drop-area");
  var fileListDiv = document.getElementById("file-list");
  var selectedVideo = document.getElementById("selected-video");

  dropArea.addEventListener("dragenter", function (e) {
      e.preventDefault();
      dropArea.style.border = "2px dashed #000";
  });

  dropArea.addEventListener("dragover", function (e) {
      e.preventDefault();
  });

  dropArea.addEventListener("dragleave", function () {
      dropArea.style.border = "2px dashed #ccc";
  });

  dropArea.addEventListener("drop", function (e) {
      e.preventDefault();
      dropArea.style.border = "2px dashed #ccc";

      var files = e.dataTransfer.files;
      handleFiles(files);
  });

  var fileInput = document.getElementById("file-input");
  fileInput.addEventListener("change", function () {
      var files = fileInput.files;
      handleFiles(files);
  });

  function displayVideo(file) {
      var reader = new FileReader();
      reader.onload = function (e) {
          selectedVideo.src = e.target.result;
      };
      reader.readAsDataURL(file);
  }

  function handleFiles(files) {
      fileListDiv.innerHTML = "";

      for (var i = 0; i < files.length; i++) {
          var file = files[i];

          if (file.type.startsWith("video/")) {
              displayVideo(file);
          } else {
              alert("Only video files are allowed.");
          }

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
});
var submitButton = document.getElementById("executeButton");
        submitButton.addEventListener("click", function () {
            // ここで選択された動画をサーバーに送信する処理を追加
            var selectedVideoFile = document.getElementById("file-input").files[0];

            if (selectedVideoFile) {
                // 仮の実装：実際のサーバー送信処理を追加してください
                alert("動画を送信しました: " + selectedVideoFile.name);
            } else {
                alert("動画ファイルが選択されていません。");
            }
        });