# -*- coding: utf-8 -*-
"""
Created on Wed May 28 18:53:29 2025

@author: tak20
"""
import cv2
import numpy as np
import os

SHOW_IMAGE = False

OUTPUT_DIR = './output' 

def warp_to_center_canvas(img, canvas_size):
    img_h, img_w = img.shape[:2]
    canvas_w, canvas_h = canvas_size
    shift_x = canvas_w / 2.0 - img_w / 2.0
    shift_y = canvas_h / 2.0 - img_h / 2.0
    M_center = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M_center, canvas_size)

def detect_and_match(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        raise ValueError("特徴点が検出されないか、ディスクリプタが空です。")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4: # 透視変換の最低条件に合わせる
        raise ValueError(f"十分なマッチング点が見つかりませんでした。現在のマッチング数: {len(matches)}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    return pts1, pts2

def estimate_transform(pts1, pts2, method):
    if method == 'translation':
        shift = np.mean(pts2 - pts1, axis=0)
        M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
        return M, 'affine'
    elif method == 'similarity':
        M, _ = cv2.estimateAffinePartial2D(pts1, pts2)
        if M is None: raise ValueError("Similarity transform estimation failed.")
        return M, 'affine'
    elif method == 'affine':
        M, _ = cv2.estimateAffine2D(pts1, pts2)
        if M is None: raise ValueError("Affine transform estimation failed.")
        return M, 'affine'
    elif method == 'perspective':
        if len(pts1) < 4 or len(pts2) < 4:
            raise ValueError("Not enough points for perspective transformation. Need at least 4.")
        H, _ = cv2.findHomography(pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
        if H is None: raise ValueError("Perspective transform estimation failed.")
        return H, 'perspective'
    else:
        raise ValueError("Unsupported transform method")

def warp_image(img, M, mode, canvas_size):
    if mode == 'affine':
        return cv2.warpAffine(img, M, canvas_size)
    else: 
        return cv2.warpPerspective(img, M, canvas_size)

def mean_image(images):
    images_float = np.array(images, dtype=np.float32)
    
    masks = np.any(images_float > 0, axis=-1) 
    
    sum_img = np.zeros_like(images_float[0], dtype=np.float32)
    count = np.zeros_like(masks[0], dtype=np.int32)
    
    for i in range(images_float.shape[0]):
        current_img = images_float[i]
        current_mask = masks[i]
        
        sum_img[current_mask] += current_img[current_mask]
        count[current_mask] += 1

    count[count == 0] = 1 
    
    mean = np.array(sum_img / count[..., np.newaxis], dtype=np.uint8)
    return mean

def crop_non_black_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        cropped = image[y:y+h, x:x+w]
        return cropped
    else:
        print("[WARN] 非黒領域が見つかりませんでした。画像全体を返します。")
        return image

def load_images():
    RESIZE_W = 432
    RESIZE_H = 432

    def read_and_resize(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"[ERROR] 画像が見つかりません: {path}")
        resized_img = cv2.resize(img, (RESIZE_W, RESIZE_H))
        
        return resized_img

    return {
        'center': read_and_resize('./image/cc.jpg'),
        'left': read_and_resize('./image/cl.jpg'),
        'right': read_and_resize('./image/cr.jpg'),
        'top': read_and_resize('./image/tc.jpg'),
        'bottom': read_and_resize('./image/bc.jpg')
    }

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] '{OUTPUT_DIR}' フォルダを作成しました。")

def save_image(path, image):
    if cv2.imwrite(path, image):
        print(f"[INFO] 保存完了: {path}")
    else:
        print(f"[ERROR] 保存失敗: {path}")

def main(transform_type, file_name):
    ensure_output_dir()
    imgs = load_images()

    canvas_size = (2000, 1500) 
    
    base_image_warped = warp_to_center_canvas(imgs['center'], canvas_size)
    warped_images = [base_image_warped]

    directions = ['left', 'right', 'top', 'bottom']
    for key in directions:
        img_to_warp = imgs[key]
        try:
            img_to_warp_gray = cv2.cvtColor(img_to_warp, cv2.COLOR_BGR2GRAY)
            base_image_warped_gray = cv2.cvtColor(base_image_warped, cv2.COLOR_BGR2GRAY)
            
            pts1, pts2 = detect_and_match(img_to_warp_gray, base_image_warped_gray)
            
            M, mode = estimate_transform(pts1, pts2, transform_type)
            warped = warp_image(img_to_warp, M, mode, canvas_size)
            
            warped_images.append(warped)
        except ValueError as ve:
            print(f"[ERROR] {key}: 変換に失敗しました - {ve}")
        except Exception as e:
            print(f"[ERROR] {key}: 予期せぬエラー - {e}")

    result = mean_image(warped_images)
    result = crop_non_black_area(result) 

    if SHOW_IMAGE:
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    save_image(f'{OUTPUT_DIR}/{file_name}', result) 

if __name__ == "__main__":
    main('translation', "translation.jpg")
    main('similarity', "similarity.jpg")
    main('affine', "affine.jpg")
    main('perspective', "perspective.jpg")