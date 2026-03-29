"""
Quantized Johnson-Lindenstrauss (QJL) による内積推定

ランダム射影 + 1ビット量子化で内積を推定する。

理論:
  ランダムベクトル s ~ N(0, I_d) に対して、
    E[sign(s^T x) * sign(s^T y)] ≈ (2/π) <x, y>
  が成り立つ（<x,y> が小さいときに良い近似）。

  したがって、m 本の独立なランダムベクトルを使って：
    <x, y> ≈ (π/2) * (1/m) Σ sign(s_k^T x) * sign(s_k^T y)
  で内積を推定できる。m → ∞ で真の値に収束する。
"""

import numpy as np


def qjl_estimate(x, y, m, rng):
    """
    QJL による内積推定（1回の試行）。

    Parameters
    ----------
    x, y : ndarray (d,)
        単位ベクトル
    m : int
        射影の本数（ランダムベクトルの数）
    rng : numpy.random.Generator
        乱数生成器

    Returns
    -------
    float
        内積の推定値
    """
    d = len(x)

    # ランダム射影行列 S ~ N(0, 1)、shape: (m, d)
    S = rng.standard_normal((m, d))

    # 射影して1ビット量子化（符号のみ保持）
    z_x = np.sign(S @ x)   # shape: (m,)
    z_y = np.sign(S @ y)   # shape: (m,)

    # 推定値 = (π/2) * (1/m) Σ z_x[k] * z_y[k]
    return (np.pi / 2) * np.mean(z_x * z_y)


def qjl_convergence(x, y, m_values, n_trials=20, seed=123):
    """
    各射影数 m に対して複数回試行し、推定値の統計量を記録する。

    射影数 m が増えるにつれて推定値のばらつきが減り、
    真の内積に収束していく様子を確認するために使う。

    Parameters
    ----------
    x, y : ndarray (d,)
        単位ベクトル
    m_values : list of int
        試す射影数のリスト
    n_trials : int
        各 m での独立な試行回数
    seed : int
        乱数シード

    Returns
    -------
    dict
        'means'  : 各 m での推定値の平均 (len(m_values),)
        'stds'   : 各 m での推定値の標準偏差 (len(m_values),)
        'all_estimates' : 全試行の推定値 (n_trials, len(m_values))
    """
    rng = np.random.default_rng(seed)
    all_estimates = np.zeros((n_trials, len(m_values)))

    for trial in range(n_trials):
        for j, m in enumerate(m_values):
            all_estimates[trial, j] = qjl_estimate(x, y, m, rng)

    return {
        "means": np.mean(all_estimates, axis=0),
        "stds": np.std(all_estimates, axis=0),
        "all_estimates": all_estimates,
    }


def qjl_incremental(x, y, max_m=500, seed=123):
    """
    射影を1本ずつ追加し、累積平均による推定値の推移を記録する。
    アニメーション用。

    1本ずつ射影を追加するたびに推定値を再計算するのではなく、
    累積和を使って効率的に計算する。

    Parameters
    ----------
    x, y : ndarray (d,)
        単位ベクトル
    max_m : int
        最大射影数
    seed : int
        乱数シード

    Returns
    -------
    estimates : list of float
        m = 1, 2, ..., max_m での累積推定値
    """
    d = len(x)
    rng = np.random.default_rng(seed)

    running_sum = 0.0
    estimates = []

    for k in range(1, max_m + 1):
        # 1本ずつランダムベクトルを生成
        s = rng.standard_normal(d)
        # 符号の積を累積
        running_sum += np.sign(s @ x) * np.sign(s @ y)
        # 現在の推定値
        estimates.append((np.pi / 2) * running_sum / k)

    return estimates


# ============================================================
# 論文版 QJL（片側量子化 — 厳密に不偏）
# ============================================================
#
# 論文では x のみを符号量子化し、y は生の射影値を使う：
#   推定値 = √(π/2) / m · Σ (sₖᵀ y) · sign(sₖᵀ x)
#
# これは全ての ⟨x,y⟩ に対して厳密に不偏:
#   E[(sᵀ y) · sign(sᵀ x)] = √(2/π) · ⟨x, y⟩
#   → √(π/2) を掛けて不偏推定量を得る
#
# 上の対称版 sign(Sx)·sign(Sy) は近似的にのみ不偏
# （⟨x,y⟩ が小さいとき良い近似）。
# ============================================================

def qjl_estimate_unbiased(x, y, m, rng):
    """
    論文版 QJL: x のみ符号量子化し、y は生の射影値を使う。

    KV キャッシュの実用場面に対応:
      - key/value は量子化して保存（sign(Sx)）
      - query は量子化せず、内積計算時に生の射影値を使用

    Parameters
    ----------
    x : ndarray (d,)
        量子化する側のベクトル（key/value に相当）
    y : ndarray (d,)
        量子化しない側のベクトル（query に相当）
    m : int
        射影数
    rng : numpy.random.Generator
        乱数生成器

    Returns
    -------
    float
        内積の推定値（厳密に不偏）
    """
    d = len(x)
    S = rng.standard_normal((m, d))

    z_x = np.sign(S @ x)    # x だけ符号量子化（1ビット）
    p_y = S @ y              # y は生の射影値（量子化しない）

    # √(π/2) / m · Σ (sₖᵀ y) · sign(sₖᵀ x)
    return np.sqrt(np.pi / 2) / m * np.dot(p_y, z_x)


def qjl_convergence_unbiased(x, y, m_values, n_trials=20, seed=123):
    """
    論文版 QJL の収束実験（複数試行）。
    """
    rng = np.random.default_rng(seed)
    all_estimates = np.zeros((n_trials, len(m_values)))

    for trial in range(n_trials):
        for j, m in enumerate(m_values):
            all_estimates[trial, j] = qjl_estimate_unbiased(x, y, m, rng)

    return {
        "means": np.mean(all_estimates, axis=0),
        "stds": np.std(all_estimates, axis=0),
        "all_estimates": all_estimates,
    }


def qjl_incremental_unbiased(x, y, max_m=500, seed=123):
    """
    論文版 QJL の累積推定値（アニメーション用）。

    射影を1本ずつ追加し、推定値が真の内積に
    厳密に収束していく様子を記録する。
    """
    d = len(x)
    rng = np.random.default_rng(seed)

    running_sum = 0.0
    estimates = []

    for k in range(1, max_m + 1):
        s = rng.standard_normal(d)
        # y は生の値、x は符号のみ
        running_sum += (s @ y) * np.sign(s @ x)
        estimates.append(np.sqrt(np.pi / 2) * running_sum / k)

    return estimates
