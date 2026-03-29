"""
TurboQuant の実装

論文 "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al., 2025) の核心アルゴリズムを教育用に実装する。

3つの Building Blocks:
  1. ランダム回転 — ベクトルの座標を i.i.d. にする前処理
  2. Lloyd-Max スカラー量子化 — 回転後の座標に適用（MSE最小化）
  3. QJL（片側量子化） — 残差に適用して内積の不偏性を確保

TurboQuant_prod（2段階アプローチ）:
  Stage 1: (b-1) ビットの MSE 量子化 → x̃_mse（再構成は良いが内積にバイアス）
  Stage 2: 残差 r = x - x̃_mse に QJL → 内積のバイアスを補正
  → 「再構成品質」と「内積の不偏性」を同時に達成
"""

import numpy as np
from .lloyd_max import lloyd_max_train, quantize


# ============================================================
# Building Block 1: ランダム回転
# ============================================================

def random_rotation_matrix(d, seed=0):
    """
    Haar 測度に従うランダム直交行列を生成する。

    ガウス乱数行列の QR 分解を利用。回転後の座標は
    近似的に i.i.d. N(0, 1/d) 分布に従い、座標ごとの
    独立なスカラー量子化が理論的に正当化される。

    Parameters
    ----------
    d : int
        次元数
    seed : int
        乱数シード

    Returns
    -------
    Q : ndarray (d, d)
        直交行列（QᵀQ = I）
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    # 符号を揃えて一意な直交行列にする
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


# ============================================================
# Building Block 2: TurboQuant MSE（ランダム回転 + Lloyd-Max）
# ============================================================

def turbo_quant_mse(x, b, rotation_seed=0, lloyd_seed=42):
    """
    TurboQuant MSE (Algorithm 1):
    ランダム回転 → Lloyd-Max スカラー量子化 → 逆回転。

    手順:
      1. y = Π · x（ランダム回転で座標を i.i.d. にする）
      2. 各 yⱼ を b ビットの Lloyd-Max で量子化 → ỹⱼ
      3. x̃ = Πᵀ · ỹ（逆回転で復元）

    Parameters
    ----------
    x : ndarray (d,)
        入力ベクトル（単位ベクトル）
    b : int
        量子化ビット幅（座標あたり）
    rotation_seed : int
        回転行列の乱数シード
    lloyd_seed : int
        Lloyd-Max 学習の乱数シード

    Returns
    -------
    x_reconstructed : ndarray (d,)
        量子化→逆量子化後のベクトル
    info : dict
        中間情報（デバッグ・可視化用）
    """
    d = len(x)

    # ランダム回転行列を生成
    Pi = random_rotation_matrix(d, seed=rotation_seed)

    # 回転：座標が ≈ i.i.d. N(0, 1/d) になる
    y = Pi @ x

    # 回転後の座標分布に合わせて Lloyd-Max を学習
    rng = np.random.default_rng(lloyd_seed)
    train_samples = rng.standard_normal(10000) / np.sqrt(d)
    n_levels = 2 ** b
    centroids, boundaries, _ = lloyd_max_train(train_samples, n_levels)

    # 各座標をスカラー量子化
    y_q = quantize(y, boundaries, centroids)

    # 逆回転で元の空間に復元
    x_reconstructed = Pi.T @ y_q

    return x_reconstructed, {
        "Pi": Pi,
        "y_rotated": y,
        "y_quantized": y_q,
        "centroids": centroids,
        "boundaries": boundaries,
    }


# ============================================================
# TurboQuant Prod: 2段階アプローチ
# ============================================================

def turbo_quant_prod_estimate(x, y_vec, b, m_qjl,
                              rotation_seed=0, lloyd_seed=42, qjl_seed=123):
    """
    TurboQuant Prod (Algorithm 2): MSE量子化 + 残差QJL。

    2段階の構成:
      Stage 1: (b-1) ビットの MSE 量子化 → x̃_mse
               内積 ⟨y, x̃_mse⟩ は再構成は良いがバイアスあり

      Stage 2: 残差 r = x - x̃_mse に QJL（片側量子化）を適用
               γ · √(π/2)/m · Σ (sₖᵀ y) · sign(sₖᵀ r)
               これが内積のバイアスを補正する

      合計: ⟨y, x̃_mse⟩ + QJL補正 → 厳密に不偏

    Parameters
    ----------
    x : ndarray (d,)
        量子化するベクトル（単位ベクトル）
    y_vec : ndarray (d,)
        内積の相手ベクトル
    b : int
        総ビット幅（b ≥ 1）
    m_qjl : int
        残差 QJL の射影数
    """
    d = len(x)

    # --- Stage 1: MSE 量子化 ---
    if b >= 2:
        x_mse, _ = turbo_quant_mse(x, b - 1, rotation_seed, lloyd_seed)
    else:
        # b=1: MSE 量子化なし（全ビットを QJL に）
        x_mse = np.zeros_like(x)

    ip_mse = float(np.dot(y_vec, x_mse))

    # --- Stage 2: 残差に QJL ---
    residual = x - x_mse
    gamma = np.linalg.norm(residual)

    # 片側量子化: 残差を符号量子化、y は生の射影値
    rng = np.random.default_rng(qjl_seed)
    S = rng.standard_normal((m_qjl, d))
    z_r = np.sign(S @ residual)    # 残差の符号量子化
    p_y = S @ y_vec                # y は量子化しない

    # QJL 補正項
    qjl_correction = float(
        gamma * np.sqrt(np.pi / 2) / m_qjl * np.dot(p_y, z_r)
    )

    # 合計（E[ip_estimate] = ⟨x, y⟩ が厳密に成立）
    ip_estimate = ip_mse + qjl_correction

    return ip_estimate, {
        "ip_mse": ip_mse,
        "qjl_correction": qjl_correction,
        "gamma": gamma,
        "residual_norm": gamma,
    }


def turbo_quant_prod_incremental(x, y_vec, b, max_m=500,
                                 rotation_seed=0, lloyd_seed=42, qjl_seed=123):
    """
    2段階アプローチの QJL 補正を逐次的に計算する（アニメーション用）。

    MSE 量子化は固定し、残差への QJL 射影を1本ずつ追加して、
    推定値が MSE のバイアス付き推定から真の内積に収束する様子を記録する。

    Returns
    -------
    estimates : list of float
        QJL 射影を k=1,2,...,max_m 本追加した時点での内積推定値
    ip_mse : float
        MSE 量子化のみの内積推定値（バイアス付き、開始点）
    """
    d = len(x)

    # Stage 1: MSE 量子化（固定）
    if b >= 2:
        x_mse, _ = turbo_quant_mse(x, b - 1, rotation_seed, lloyd_seed)
    else:
        x_mse = np.zeros_like(x)

    ip_mse = float(np.dot(y_vec, x_mse))
    residual = x - x_mse
    gamma = np.linalg.norm(residual)

    # Stage 2: QJL を1本ずつ追加
    rng = np.random.default_rng(qjl_seed)
    running_sum = 0.0
    estimates = []

    for k in range(1, max_m + 1):
        s = rng.standard_normal(d)
        running_sum += (s @ y_vec) * np.sign(s @ residual)
        qjl_correction = gamma * np.sqrt(np.pi / 2) * running_sum / k
        estimates.append(ip_mse + qjl_correction)

    return estimates, ip_mse
