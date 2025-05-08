import numpy as np
import cv2 as cv
import glob

# チェスボードパターンサイズ（内角数）
pattern_size = (7, 11)

# 3Dオブジェクトポイント（チェスボード上の点群）
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('*.jpg')

first_corners_image_saved = False

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        img_corners = cv.drawChessboardCorners(img.copy(), pattern_size, corners2, ret)

        if not first_corners_image_saved:
            cv.imwrite("corners.jpg", img_corners)
            first_corners_image_saved = True

cv.destroyAllWindows()

# Zhangの方法でカメラキャリブレーション
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("=== Intrinsic Parameters (Camera Matrix) ===\n", mtx)
print("\n=== Distortion Coefficients ===\n", dist.ravel())

# Rotation, Translation ベクトルのNumPy配列まとめ
rvec_array = np.array(rvecs).reshape(-1, 3, 1)
tvec_array = np.array(tvecs).reshape(-1, 3, 1)

print("\n=== Rotation Vectors ===\n", rvec_array)
print("\n=== Translation Vectors ===\n", tvec_array)

# リプロジェクション誤差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error
print("\n=== Mean Reprojection Error ===")
print(total_error / len(objpoints))

# --- 3D立体（立方体）の投影描画（最初の1枚だけ） ---

cube_size = 1.0
cube = np.float32([
    [0, 0, 0], [0, cube_size, 0], [cube_size, cube_size, 0], [cube_size, 0, 0],
    [0, 0, -cube_size], [0, cube_size, -cube_size], [cube_size, cube_size, -cube_size], [cube_size, 0, -cube_size]
])

def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)
    for i in range(4):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (255, 0, 0), 3)
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

first_cube_saved = False

for i, fname in enumerate(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if ret:
        imgpts, _ = cv.projectPoints(cube, rvecs[i], tvecs[i], mtx, dist)
        img_cube = draw_cube(img.copy(), imgpts)

        if not first_cube_saved:
            cv.imwrite("cube.jpg", img_cube)
            first_cube_saved = True

        break  # 最初の一枚だけ描画
