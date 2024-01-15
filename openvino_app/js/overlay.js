document.addEventListener("DOMContentLoaded", function () {
  var overlayButton = document.getElementById("overlayButton");
  var overlay = document.getElementById("overlay");
  var closeOverlayButton = document.getElementById("closeOverlayButton");

  overlayButton.addEventListener("click", function () {
    overlay.style.display = "flex";
    setTimeout(function () {
      var result = confirm(
        "推論が完了しました。結果ページに移動します。キャンセルすると、推論結果は破棄され最初から開始されます。"
      );
      if (result) {
        window.location.href = "result.html";
      }
    }, 5000);
  });

  closeOverlayButton.addEventListener("click", function () {
    overlay.style.display = "none";
  });
});
