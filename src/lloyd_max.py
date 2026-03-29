"""
Lloyd-Max MSE最小化スカラー量子化器

目的関数: min E[(X - Q(X))^2]
つまり、量子化後の再構成誤差（MSE）を最小化する。

アルゴリズム:
  1. セントロイド（再構成値）を初期化
  2. 境界を隣接セントロイドの中点に更新
  3. セントロイドを各区間に属するサンプルの条件付き平均に更新
  4. MSE が収束するまで 2-3 を繰り返す

注意:
  MSE最小化は「元の値の再構成」に最適であり、
  「内積の保存」には最適でない。この違いがQJLとの
  比較で重要なポイントとなる。
"""

import numpy as np


def lloyd_max_train(samples, n_levels=4, max_iter=50, tol=1e-8):
    """
    Lloyd-Max アルゴリズムで量子化器を学習する。

    Parameters
    ----------
    samples : ndarray (n,)
        学習用サンプル
    n_levels : int
        量子化レベル数（セントロイドの数）
    max_iter : int
        最大反復回数
    tol : float
        MSE変化の収束閾値

    Returns
    -------
    centroids : ndarray (n_levels,)
        最終的なセントロイド位置
    boundaries : ndarray (n_levels - 1,)
        最終的な決定境界
    history : list of dict
        各反復のスナップショット
        - 'iteration': 反復番号
        - 'centroids': セントロイド位置
        - 'boundaries': 境界位置
        - 'mse': その反復でのMSE
    """
    # --- 初期化 ---
    # サンプルの分位点でセントロイドを初期化する。
    # 例: n_levels=4 なら [0.2, 0.4, 0.6, 0.8] 分位点を使用。
    quantiles = np.linspace(0, 1, n_levels + 2)[1:-1]
    centroids = np.quantile(samples, quantiles)

    history = []
    prev_mse = float("inf")

    for iteration in range(max_iter):
        # --- Step 1: 境界の更新 ---
        # 隣接するセントロイドの中点を新しい境界とする。
        # サンプルはこの境界より近いセントロイドに割り当てられる。
        boundaries = (centroids[:-1] + centroids[1:]) / 2

        # --- Step 2: サンプルの割り当て ---
        # np.digitize: 各サンプルがどの区間に属するかを決定
        # 区間 i: boundaries[i-1] < sample <= boundaries[i]
        assignments = np.digitize(samples, boundaries)

        # --- Step 3: セントロイドの更新 ---
        # 各区間に割り当てられたサンプルの平均を新しいセントロイドとする。
        # これが条件付き期待値 E[X | X ∈ 区間 i] に相当する。
        new_centroids = np.empty_like(centroids)
        for i in range(n_levels):
            mask = assignments == i
            if np.sum(mask) > 0:
                new_centroids[i] = np.mean(samples[mask])
            else:
                # 空の区間が発生した場合は前の値を保持
                new_centroids[i] = centroids[i]

        centroids = new_centroids

        # --- MSE の計算 ---
        quantized = centroids[assignments]
        mse = np.mean((samples - quantized) ** 2)

        history.append({
            "iteration": iteration,
            "centroids": centroids.copy(),
            "boundaries": boundaries.copy(),
            "mse": mse,
        })

        # 収束判定
        if abs(prev_mse - mse) < tol:
            break
        prev_mse = mse

    return centroids, boundaries, history


def quantize(values, boundaries, centroids):
    """
    学習済みの量子化器で値を量子化する。

    各値を境界に基づいて最近傍のセントロイドに置き換える。

    Parameters
    ----------
    values : ndarray
        量子化する値（任意の shape）
    boundaries : ndarray (n_levels - 1,)
        決定境界
    centroids : ndarray (n_levels,)
        セントロイド（再構成値）

    Returns
    -------
    ndarray
        量子化された値（入力と同じ shape）
    """
    assignments = np.digitize(values, boundaries)
    return centroids[assignments]
