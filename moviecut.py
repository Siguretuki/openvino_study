from moviepy.video.io.VideoFileClip import VideoFileClip

def cut_video(input_file, output_file, duration):
    # 動画ファイルを読み込む
    video_clip = VideoFileClip(input_file)

    # 指定した秒数で動画を切り取る
    remaining_clip = video_clip.subclip(duration)

    # 残りの動画を保存する
    remaining_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    # 入力ファイルのパス
    input_file_path = "./uploads/test.mp4"

    # 出力ファイルのパス
    output_file_path = "./uploads/test2.mp4"

    # 切り取りたい秒数をテキスト入力で受け取る
    duration = float(input("残したい秒数を入力してください: "))

    # 動画を切り取る
    cut_video(input_file_path, output_file_path, duration)
