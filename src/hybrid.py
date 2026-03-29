"""
ハイブリッド手法：ランダム射影 + 低ビット量子化

QJL（1ビット）と通常のランダム射影（float32）の間を補間する手法。
射影後の値を b ビットに量子化することで、メモリ使用量と内積精度の
トレードオフを制御する。

  b = 1 の場合: QJL と同等（符号のみ保持、π/2 補正）
  b ≥ 2 の場合: 均一量子化で射影値の大きさも部分的に保持

メモリ使用量: m × b ビット／ベクトル
"""

import numpy as np


def uniform_quantize(values, b, val_range=(-3.0, 3.0)):
    """
    対称均一量子化器。

    値を [vmin, vmax] の範囲で 2^b レベルに量子化する。
    範囲外の値はクリッピングされる。

    Parameters
    ----------
    values : ndarray
        量子化する値
    b : int
        ビット数（2^b レベル）
    val_range : tuple (float, float)
        量子化範囲 (vmin, vmax)

    Returns
    -------
    ndarray
        量子化された値（入力と同じ shape）
    """
    n_levels = 2 ** b
    vmin, vmax = val_range

    # 範囲外をクリッピング
    clipped = np.clip(values, vmin, vmax)

    # [0, 1] に正規化 → n_levels 個のレベルに丸め → 元のスケールに戻す
    normalized = (clipped - vmin) / (vmax - vmin)
    quantized_idx = np.round(normalized * (n_levels - 1))
    quantized = quantized_idx / (n_levels - 1) * (vmax - vmin) + vmin

    return quantized


def hybrid_estimate(x, y, m, b, rng, val_range=(-3.0, 3.0)):
    """
    ハイブリッド手法による内積推定。

    手順:
      1. ランダム射影行列 S ~ N(0,1) を生成（shape: m × d）
      2. 射影: p_x = S @ x,  p_y = S @ y（各 m 次元）
      3. b ビット量子化: p_x_q = Q_b(p_x),  p_y_q = Q_b(p_y)
      4. 内積推定: mean(p_x_q * p_y_q)

    b = 1 の場合は符号量子化 + π/2 補正（QJL と同等）。
    b ≥ 2 の場合は均一量子化で大きさも保持する。

    射影値は N(0, 1) 分布（単位ベクトルの場合）なので、
    val_range=(-3, 3) で 99.7% をカバーする。

    Parameters
    ----------
    x, y : ndarray (d,)
        単位ベクトル
    m : int
        射影数
    b : int
        量子化ビット数
    rng : numpy.random.Generator
        乱数生成器
    val_range : tuple
        量子化範囲

    Returns
    -------
    float
        内積の推定値
    """
    d = len(x)
    S = rng.standard_normal((m, d))

    # ランダム射影
    p_x = S @ x   # shape: (m,)
    p_y = S @ y   # shape: (m,)

    if b == 1:
        # 1ビット: 符号量子化 + π/2 補正（QJL と同等）
        p_x_q = np.sign(p_x)
        p_y_q = np.sign(p_y)
        return (np.pi / 2) * np.mean(p_x_q * p_y_q)
    else:
        # 多ビット: 均一量子化（大きさの情報を部分的に保持）
        p_x_q = uniform_quantize(p_x, b, val_range=val_range)
        p_y_q = uniform_quantize(p_y, b, val_range=val_range)
        return np.mean(p_x_q * p_y_q)
