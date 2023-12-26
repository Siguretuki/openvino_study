import cv2

# カメラを起動
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラを使用することを示します

# 動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4形式を使用する場合
width, height = 640, 480  # 画像の幅と高さを設定
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))  # 出力ファイルの設定

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
