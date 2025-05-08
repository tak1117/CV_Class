# -*- coding: utf-8 -*-
"""
Created on Thu May  8 23:33:59 2025

@author: tak20
"""
import numpy as np
import cv2

# グローバル変数
pts1 = []
pts2 = np.float32([[100, 200], [500, 200], [100, 700], [500, 700]])

# マウスイベントのコールバック関数
def click_event(event, x, y, flags, param):
    global pts1, img
    if event == cv2.EVENT_LBUTTONDOWN:  # 左クリック
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # クリックした位置に赤い点を描く
        pts1.append([x, y])  # クリック位置をpts1に追加
        cv2.imshow("input", img)  # 変更された画像を表示
        
        # 4つのポイントを選んだら変換を実行
        if len(pts1) == 4:
            pts1_np = np.float32(pts1)
            M = cv2.getPerspectiveTransform(pts1_np, pts2)
            dst = cv2.warpPerspective(img, M, (600, 800))
            cv2.imshow('output', dst)
            cv2.imwrite('rectified.jpg', dst)

# 画像を読み込む
img = cv2.imread("sagrada.jpg")

# 画像にマウスイベントを設定
cv2.imshow("input", img)
cv2.setMouseCallback("input", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
