from flask import Flask, render_template, request, send_file
import os

app = Flask(__name__)

# アップロードされた動画ファイルを保存するディレクトリ
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ホームページ
@app.route('/')
def home():
    return render_template('index.html')

# 動画のアップロード処理
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # ファイルを保存
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return "File uploaded successfully"

# 動画のダウンロード処理
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    # サーバーを起動
    app.run(debug=True)
