# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:18:33 2025

@author: tak20
"""

import cv2
import math
import numpy as np
import cv2.ximgproc as xip
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === 入力画像読み込み & 前処理 ===
img = cv2.imread("input.jpg")
img = cv2.resize(img, (512, 512))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ノイズ付加
mean, sigma = 0, 25
noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)

# === パラメータセット ===
param_bilateral = [
    (6, 75, 75),
    (12, 75, 75),
    (6, 150, 75),
    (6, 75, 150),
]

param_guided = [
    (16, 1e-2),
    (8, 1e-2),
    (16, 1e-4),
]

# === 結果格納用 ===
results = [("Original", img_rgb), ("Noisy", noisy_rgb)]

# === Bilateral 処理 ===
for d, sigmaColor, sigmaSpace in param_bilateral:
    out = cv2.bilateralFilter(noisy_img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    label = f"Bilateral d={d} sc={sigmaColor} ss={sigmaSpace}"
    results.append((label, out_rgb))

# === Guided 処理 ===
for r, e in param_guided:
    out = xip.guidedFilter(guide=img, src=noisy_img, radius=r, eps=e, dDepth=-1)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    label = f"Guided r={r} eps={e:.0e}"
    results.append((label, out_rgb))


# === デフォルト結果も保存 ===
bilateral_default = cv2.bilateralFilter(noisy_img, 9, 75, 75)
guided_default = xip.guidedFilter(guide=img, src=noisy_img, radius=16, eps=1e-2, dDepth=-1)


# === デフォルト比較可視化 ===
plt.figure(figsize=(10, 8))
titles = ["Original", "Noisy", "Bilateral Default", "Guided Default"]
images = [img_rgb, noisy_rgb,
          cv2.cvtColor(bilateral_default, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(guided_default, cv2.COLOR_BGR2RGB)]
for i in range(4):
    plt.subplot(2, 2, i + 1)  # ← 2行×2列に変更
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("default_filter_comparison.png")
plt.show()


# === 定量評価 ===
labels, scores_psnr, scores_ssim = [], [], []
for name, image in results:
    p = psnr(img_rgb, image, data_range=255)
    s = ssim(img_rgb, image, data_range=255, channel_axis=2)
    labels.append(name)
    scores_psnr.append(p)
    scores_ssim.append(s)
    print(f"{name}:\n  PSNR: {p:.2f} dB\n  SSIM: {s:.4f}")

# === 全体：PSNR / SSIM 別グラフ ===
#x = np.arange(len(labels))

# === Bilateral だけの抽出とグラフ・画像 ===
bilateral_subset = [(l, img) for (l, img) in results if l.startswith("Bilateral")]
b_labels = [l for l, _ in bilateral_subset]
b_imgs = [img for _, img in bilateral_subset]
b_psnr = [scores_psnr[labels.index(l)] for l in b_labels]
b_ssim = [scores_ssim[labels.index(l)] for l in b_labels]

# 画像比較
n = len(bilateral_subset)
cols = 2
rows = math.ceil(n / cols)

plt.figure(figsize=(6 * cols, 4 * rows))
for i, (l, img) in enumerate(bilateral_subset):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(l)
    plt.axis('off')
plt.tight_layout()
plt.savefig("bilateral_only_comparison.png")
plt.show()

# グラフ
x = np.arange(len(b_labels))
plt.figure(figsize=(10, 5))
plt.bar(x, b_psnr, color='cornflowerblue')
plt.xticks(x, b_labels, rotation=45, ha='right')
plt.ylabel('PSNR (dB)')
plt.title('Bilateral Filter: PSNR Comparison')
plt.tight_layout()
plt.savefig("bilateral_psnr.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(x, b_ssim, color='mediumseagreen')
plt.xticks(x, b_labels, rotation=45, ha='right')
plt.ylabel('SSIM')
plt.title('Bilateral Filter: SSIM Comparison')
plt.tight_layout()
plt.savefig("bilateral_ssim.png")
plt.show()

# === Guided だけの抽出とグラフ・画像 ===
guided_subset = [(l, img) for (l, img) in results if l.startswith("Guided")]
g_labels = [l for l, _ in guided_subset]
g_imgs = [img for _, img in guided_subset]
g_psnr = [scores_psnr[labels.index(l)] for l in g_labels]
g_ssim = [scores_ssim[labels.index(l)] for l in g_labels]

# 画像比較
n = len(guided_subset)
cols = 2
rows = math.ceil(n / cols)

plt.figure(figsize=(6 * cols, 4 * rows))
for i, (l, img) in enumerate(guided_subset):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(l)
    plt.axis('off')
plt.tight_layout()
plt.savefig("guided_only_comparison.png")
plt.show()

# グラフ
x = np.arange(len(g_labels))
plt.figure(figsize=(10, 5))
plt.bar(x, g_psnr, color='steelblue')
plt.xticks(x, g_labels, rotation=45, ha='right')
plt.ylabel('PSNR (dB)')
plt.title('Guided Filter: PSNR Comparison')
plt.tight_layout()
plt.savefig("guided_psnr.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(x, g_ssim, color='seagreen')
plt.xticks(x, g_labels, rotation=45, ha='right')
plt.ylabel('SSIM')
plt.title('Guided Filter: SSIM Comparison')
plt.tight_layout()
plt.savefig("guided_ssim.png")
plt.show()

# === ノイズ + Bilateral（デフォルト）+ Guided（デフォルト） を比較するグラフ ===

# 実際のラベル名に一致するように変更
compare_labels = [
    "Noisy",
    "Bilateral d=6 sc=75 ss=75",     # 使用したBilateralのラベル
    "Guided r=16 eps=1e-02"           # 使用したGuidedのラベル
]

# PSNRとSSIMの抽出
compare_psnr = [scores_psnr[labels.index(l)] for l in compare_labels]
compare_ssim = [scores_ssim[labels.index(l)] for l in compare_labels]

# PSNR グラフ
x = np.arange(len(compare_labels))
plt.figure(figsize=(8, 5))
plt.bar(x, compare_psnr, color=['gray', 'cornflowerblue', 'steelblue'])
plt.xticks(x, compare_labels)
plt.ylabel('PSNR (dB)')
plt.title('PSNR Comparison (Noisy vs. Bilateral vs. Guided)')
plt.tight_layout()
plt.savefig("psnr_noisy_bilateral_guided.png")
plt.show()

# SSIM グラフ
plt.figure(figsize=(8, 5))
plt.bar(x, compare_ssim, color=['gray', 'mediumseagreen', 'seagreen'])
plt.xticks(x, compare_labels)
plt.ylabel('SSIM')
plt.title('SSIM Comparison (Noisy vs. Bilateral vs. Guided)')
plt.tight_layout()
plt.savefig("ssim_noisy_bilateral_guided.png")
plt.show()

