import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像読み込み
img1 = cv2.imread('img1.jpg', 0)  # 左画像
img2 = cv2.imread('img2.jpg', 0)  # 右画像

if img1 is None or img2 is None:
    raise FileNotFoundError("画像が見つかりません。ファイルパスを確認してください。")

# SIFT 初期化
sift = cv2.SIFT_create()

# 特徴点検出と記述
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN パラメータ
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# マッチング
matches = flann.knnMatch(des1, des2, k=2)
good = []
pts1 = []
pts2 = []

for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# 基本行列推定
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# --- ここで Fundamental Matrix を出力 ---
print("Fundamental Matrix (F):\n", F)

# インライアのみ抽出
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# エピポーラ線描画関数
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(pt1), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(pt2), 5, color, -1)
    return img1_color, img2_color

# エピポーラ線の計算と描画
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

# 画像を保存
cv2.imwrite('epilines_left.jpg', img5)
cv2.imwrite('epilines_right.jpg', img3)

# 表示（OpenCV -> Matplotlib形式に変換）
plt.subplot(121)
plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
plt.title("Epilines on Left Image")
plt.xlabel("Pixel (X)")
plt.ylabel("Pixel (Y)")

plt.subplot(122)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.title("Epilines on Right Image")
plt.xlabel("Pixel (X)")
plt.ylabel("Pixel (Y)")

plt.tight_layout()
plt.show()

