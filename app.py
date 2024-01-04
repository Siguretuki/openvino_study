from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for,send_from_directory
import os
from werkzeug.utils import secure_filename
import datetime

app = Flask(__name__, static_folder=None)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    filenames = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', filenames=filenames)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'})

    # 前の動画を削除
    previous_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
    if os.path.exists(previous_video_path):
        os.remove(previous_video_path)

    # 新しい動画を保存
    filename = secure_filename('video.mp4')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return redirect(url_for('result')) #jsonify({'message': 'File uploaded successfully', 'file_path': file_path})

@app.route('/uploads/<path:filename>')
def serve_video(filename):
    return send_file('uploads/' + filename, mimetype='video/mp4')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/execute', methods=['GET'])
def execute_python_file():
    # 実際の処理はここに追加
    # 例：os.system('python your_script.py')
    
    return jsonify({'message': 'Python file executed successfully'})

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/static/<path:filename>')
def static(filename):
    directory_path = os.path.join(app.root_path, 'static')

    last_modified = None
    if '.css' in filename:
        last_modified = datetime.datetime.now(datetime.timezone.utc)

    max_age = app.get_send_file_max_age(filename)
    return send_from_directory(
        directory_path, filename, last_modified=last_modified, max_age=max_age
    )

# 以前のupload_file()を削除
# 以前のdownload_file()を削除

if __name__ == '__main__':
    app.run(debug=True)
