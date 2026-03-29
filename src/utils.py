"""
ユーティリティ関数

ベクトル生成、サンプル生成、メモリ計算など共通で使う機能を提供する。
"""

import numpy as np


def generate_vectors(d=100, seed=42):
    """
    次元 d のランダム単位ベクトル x, y を生成する。

    Parameters
    ----------
    d : int
        ベクトルの次元数
    seed : int
        乱数シード（再現性のため）

    Returns
    -------
    x : ndarray (d,)
        L2正規化された単位ベクトル
    y : ndarray (d,)
        L2正規化された単位ベクトル
    true_ip : float
        真の内積（= cosine similarity、単位ベクトルなので一致）
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(d)
    y = rng.standard_normal(d)

    # L2正規化：||x|| = ||y|| = 1 にする
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    true_ip = np.dot(x, y)
    return x, y, true_ip


def generate_vectors_with_ip(d=100, target_ip=0.5, seed=42):
    """
    指定した内積を持つ単位ベクトルのペアを生成する。

    Parameters
    ----------
    d : int
        次元数
    target_ip : float
        目標の内積（-1 < target_ip < 1）
    seed : int
        乱数シード

    Returns
    -------
    x, y : ndarray (d,)
        単位ベクトル（⟨x, y⟩ ≈ target_ip）
    true_ip : float
        実際の内積
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(d)
    x = x / np.linalg.norm(x)

    # x と直交する成分を作る
    z = rng.standard_normal(d)
    z = z - np.dot(z, x) * x
    z = z / np.linalg.norm(z)

    # y = target_ip * x + sqrt(1 - target_ip^2) * z
    y = target_ip * x + np.sqrt(1 - target_ip ** 2) * z
    y = y / np.linalg.norm(y)

    true_ip = float(np.dot(x, y))
    return x, y, true_ip


def generate_samples(n=10000, distribution="normal", seed=42):
    """
    指定された分布からサンプルを生成する。
    Lloyd-Max 量子化器の学習に使用。

    Parameters
    ----------
    n : int
        サンプル数
    distribution : str
        分布の種類（'normal': 標準正規分布）
    seed : int
        乱数シード

    Returns
    -------
    samples : ndarray (n,)
    """
    rng = np.random.default_rng(seed)

    if distribution == "normal":
        return rng.standard_normal(n)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def compute_memory_bits(method, d=None, m=None, b=None):
    """
    各手法のメモリ使用量（ビット数/ベクトル）を計算する。

    Parameters
    ----------
    method : str
        'baseline', 'lloyd_max', 'qjl', 'multibit_rp'
    d : int or None
        ベクトルの次元数（baseline, lloyd_max で必要）
    m : int or None
        射影数（qjl, multibit_rp で必要）
    b : int or None
        ビット数（lloyd_max, multibit_rp で必要）

    Returns
    -------
    int
        ビット数

    メモリ計算の根拠:
      - Baseline (float32): 32 × d ビット
      - Lloyd-Max b-bit:     b × d ビット
      - QJL m 射影:          m × 1 ビット（各射影は符号1ビット）
      - Multi-bit RP m 射影 b-bit: m × b ビット
    """
    if method == "baseline":
        return 32 * d
    elif method == "lloyd_max":
        return b * d
    elif method == "qjl":
        return m
    elif method == "multibit_rp":
        return m * b
    else:
        raise ValueError(f"Unknown method: {method}")
