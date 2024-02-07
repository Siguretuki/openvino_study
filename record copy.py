import cv2
from tkinter import filedialog

file=filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])

cap = cv2.VideoCapture(file)

# 動画の設定
fourcc = cv2.VideoWriter_fourcc(*'h264') 
width, height = 640, 480  # 画像の幅と高さを設定
out = cv2.VideoWriter('result.mp4', fourcc, 30.0, (width, height))  # 出力ファイルの設定

while True:
    # カメラからフレームを取得
    ret, frame = cap.read()

    # フレームが正しく取得できた場合
    if ret:
        # フレームを表示する場合（任意）
        cv2.imshow('Camera', frame)

        # 動画にフレームを書き込み
        out.write(frame)

        # 'q'を押すとループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# カメラとファイル出力を解放
cap.release()
out.release()

# ウィンドウを閉じる
cv2.destroyAllWindows()
