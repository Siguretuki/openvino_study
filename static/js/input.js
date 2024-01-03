document.getElementById('file-input').addEventListener('change', function() {
    var fileInput = document.getElementById('file-input');
    var selectedVideo = document.getElementById('selected-video');
    
    var file = fileInput.files[0];

    // サーバーにファイルをアップロードする
    var formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      // サーバーからの応答を表示
      document.getElementById('file-list').innerText = data.message;

      // アップロードされた動画を再生
      selectedVideo.src = '/download/' + file.name;
    })
    .catch(error => {
      console.error('Error:', error);
    });
  });

  document.getElementById('execute-button').addEventListener('click', function() {
    // サーバーにPythonファイルを実行するリクエストを送信
    fetch('/execute')
    .then(response => response.json())
    .then(data => {
      // サーバーからの応答を表示
      alert(data.message);
    })
    .catch(error => {
      console.error('Error:', error);
    });
  });