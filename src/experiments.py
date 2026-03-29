"""
実験モジュール

5つの手法（Baseline、Lloyd-Max、QJL、Multi-bit RP、TurboQuant）の実験を実行し、
内積誤差・再構成誤差・メモリ使用量を計算して構造化された結果を返す。
"""

import numpy as np

from .utils import compute_memory_bits
from .qjl import qjl_convergence, qjl_convergence_unbiased
from .lloyd_max import lloyd_max_train, quantize
from .multibit_rp import multibit_rp_estimate
from .turbo_quant import turbo_quant_prod_estimate


# ============================================================
# 個別実験
# ============================================================

def run_baseline(x, y, d):
    """
    Baseline: float32 のベクトルでそのまま内積を計算する。

    全ての誤差の基準となる。メモリ使用量は 32 × d ビット。
    """
    true_ip = float(np.dot(x, y))
    return {
        "method": "Baseline",
        "config": "float32",
        "inner_product": true_ip,
        "ip_error": 0.0,
        "reconstruction_mse": 0.0,
        "memory_bits": compute_memory_bits("baseline", d=d),
    }


def run_lloyd_max_experiments(x, y, d, n_levels_list=None, seed=42):
    """
    Lloyd-Max: 各ビット数で量子化し、内積と再構成MSEを計算する。

    量子化器はベクトル成分の分布 ≈ N(0, 1/√d) で学習する。
    量子化後のベクトル間のコサイン類似度を内積推定値とする。

    Parameters
    ----------
    x, y : ndarray
        単位ベクトル
    d : int
        次元数
    n_levels_list : list of int
        量子化レベル数のリスト（各要素は 2 の冪）
    seed : int
        乱数シード
    """
    if n_levels_list is None:
        n_levels_list = [2, 4, 8, 16, 32, 64, 128, 256]

    true_ip = float(np.dot(x, y))

    # 単位ベクトルの成分分布で学習データを生成
    rng = np.random.default_rng(seed + 200)
    train_samples = rng.standard_normal(10000) / np.sqrt(d)

    results = []
    for n_levels in n_levels_list:
        bits = int(np.log2(n_levels))
        centroids, boundaries, _ = lloyd_max_train(train_samples, n_levels)

        # 各成分を量子化
        x_q = quantize(x, boundaries, centroids)
        y_q = quantize(y, boundaries, centroids)

        # コサイン類似度で内積を推定
        norm_xq = np.linalg.norm(x_q)
        norm_yq = np.linalg.norm(y_q)
        if norm_xq > 0 and norm_yq > 0:
            cos_sim = float(np.dot(x_q, y_q) / (norm_xq * norm_yq))
        else:
            cos_sim = 0.0

        # 再構成MSE
        recon_mse = float((np.mean((x - x_q) ** 2) + np.mean((y - y_q) ** 2)) / 2)

        results.append({
            "method": "Lloyd-Max",
            "config": f"{bits}-bit",
            "n_levels": n_levels,
            "bits": bits,
            "inner_product": cos_sim,
            "ip_error": abs(cos_sim - true_ip),
            "reconstruction_mse": recon_mse,
            "memory_bits": compute_memory_bits("lloyd_max", d=d, b=bits),
        })

    return results


def run_qjl_experiments(x, y, m_values=None, n_trials=30, seed=123):
    """
    QJL: 各射影数で複数回試行し、内積推定の統計量を計算する。

    Parameters
    ----------
    x, y : ndarray
        単位ベクトル
    m_values : list of int
        射影数のリスト
    n_trials : int
        各 m での独立な試行回数
    seed : int
        乱数シード

    Returns
    -------
    results : list of dict
        各 m での結果
    conv_data : dict
        qjl_convergence() の生データ（収束グラフ用）
    """
    if m_values is None:
        m_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    true_ip = float(np.dot(x, y))
    conv_data = qjl_convergence(x, y, m_values, n_trials=n_trials, seed=seed)

    results = []
    for j, m in enumerate(m_values):
        estimates = conv_data["all_estimates"][:, j]
        errors = np.abs(estimates - true_ip)

        results.append({
            "method": "QJL",
            "config": f"m={m}",
            "m": m,
            "ip_mean": float(conv_data["means"][j]),
            "ip_std": float(conv_data["stds"][j]),
            "ip_error_mean": float(np.mean(errors)),
            "reconstruction_mse": None,   # QJL では再構成不可
            "memory_bits": compute_memory_bits("qjl", m=m),
        })

    return results, conv_data


def run_multibit_rp_experiments(x, y, configs=None, n_trials=30, seed=456):
    """
    Multi-bit RP: 各 (m, b) 設定で内積推定の統計量を計算する。

    Parameters
    ----------
    x, y : ndarray
        単位ベクトル
    configs : list of (int, int)
        (射影数 m, ビット数 b) のリスト
    n_trials : int
        各設定での試行回数
    seed : int
        乱数シード
    """
    if configs is None:
        configs = [
            # b=2: 各射影を 2ビットで量子化
            (10, 2), (20, 2), (50, 2), (100, 2), (200, 2), (500, 2),
            # b=4: 各射影を 4ビットで量子化
            (10, 4), (20, 4), (50, 4), (100, 4), (200, 4), (500, 4),
        ]

    true_ip = float(np.dot(x, y))

    results = []
    for m, b in configs:
        estimates = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial * 1000 + m * 10 + b)
            est = multibit_rp_estimate(x, y, m, b, rng)
            estimates.append(est)

        estimates = np.array(estimates)
        errors = np.abs(estimates - true_ip)

        results.append({
            "method": "Multi-bit RP",
            "config": f"m={m},b={b}",
            "m": m,
            "b": b,
            "ip_mean": float(np.mean(estimates)),
            "ip_std": float(np.std(estimates)),
            "ip_error_mean": float(np.mean(errors)),
            "reconstruction_mse": None,   # 再構成不可
            "memory_bits": compute_memory_bits("multibit_rp", m=m, b=b),
        })

    return results


def run_turbo_quant_experiments(x, y, d, b_values=None, m_qjl_values=None,
                                n_trials=30, seed=789):
    """
    TurboQuant Prod（2段階: MSE量子化 + 残差QJL）の実験。

    各 (b, m_qjl) の組み合わせで内積推定精度を評価する。
    b ビットの内訳: (b-1) ビット MSE + m_qjl 本の QJL 射影。
    """
    if b_values is None:
        b_values = [2, 3, 4]
    if m_qjl_values is None:
        m_qjl_values = [50, 100, 200]

    true_ip = float(np.dot(x, y))

    results = []
    for b in b_values:
        for m_qjl in m_qjl_values:
            estimates = []
            for trial in range(n_trials):
                est, _ = turbo_quant_prod_estimate(
                    x, y, b, m_qjl,
                    rotation_seed=seed + trial,
                    lloyd_seed=seed + 100,
                    qjl_seed=seed + trial * 1000 + m_qjl,
                )
                estimates.append(est)

            estimates = np.array(estimates)
            errors = np.abs(estimates - true_ip)
            # メモリ: (b-1)*d ビット (MSE) + m_qjl ビット (QJL)
            memory = (b - 1) * d + m_qjl

            results.append({
                "method": "TurboQuant",
                "config": f"b={b},m={m_qjl}",
                "b": b,
                "m_qjl": m_qjl,
                "ip_mean": float(np.mean(estimates)),
                "ip_std": float(np.std(estimates)),
                "ip_error_mean": float(np.mean(errors)),
                "memory_bits": memory,
                "reconstruction_mse": None,
            })

    return results


# ============================================================
# 全実験の統合
# ============================================================

def run_all_experiments(x, y, d, seed=42):
    """
    4つの手法の実験を全て実行し、結果を辞書にまとめて返す。

    Parameters
    ----------
    x, y : ndarray
        単位ベクトル
    d : int
        次元数
    seed : int
        乱数シード

    Returns
    -------
    dict
        'd': 次元数
        'true_ip': 真の内積
        'baseline': Baseline の結果
        'lloyd_max': Lloyd-Max の結果リスト
        'qjl': QJL の結果リスト
        'multibit_rp': Multi-bit RP の結果リスト
        'qjl_conv_data': QJL 収束データ（グラフ用）
        'm_values': QJL/グラフ用の射影数リスト
    """
    m_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    print("  Running Baseline...")
    baseline = run_baseline(x, y, d)

    print("  Running Lloyd-Max experiments...")
    lloyd_max = run_lloyd_max_experiments(x, y, d, seed=seed)

    print("  Running QJL experiments...")
    qjl, qjl_conv_data = run_qjl_experiments(
        x, y, m_values=m_values, n_trials=30, seed=seed + 1
    )

    print("  Running Multi-bit RP experiments...")
    multibit_rp = run_multibit_rp_experiments(x, y, seed=seed + 2)

    print("  Running TurboQuant experiments...")
    turbo_quant = run_turbo_quant_experiments(x, y, d, seed=seed + 3)

    # 論文版 QJL（片側量子化）の収束データ
    print("  Running Paper QJL (unbiased) experiments...")
    qjl_conv_unbiased = qjl_convergence_unbiased(
        x, y, m_values, n_trials=30, seed=seed + 4
    )

    return {
        "d": d,
        "true_ip": float(np.dot(x, y)),
        "baseline": baseline,
        "lloyd_max": lloyd_max,
        "qjl": qjl,
        "multibit_rp": multibit_rp,
        "turbo_quant": turbo_quant,
        "qjl_conv_data": qjl_conv_data,
        "qjl_conv_unbiased": qjl_conv_unbiased,
        "m_values": m_values,
    }


def print_summary_table(results):
    """
    実験結果のサマリを表形式でコンソールに出力する。

    Parameters
    ----------
    results : dict
        run_all_experiments() の戻り値
    """
    true_ip = results["true_ip"]

    print()
    header = f"{'Method':<12} {'Config':<14} {'Memory':>10} {'IP Error':>12} {'Recon MSE':>12}"
    print(header)
    print("-" * len(header))

    # Baseline
    b = results["baseline"]
    print(f"{'Baseline':<12} {'float32':<14} {b['memory_bits']:>8} b  {b['ip_error']:>11.6f}  {b['reconstruction_mse']:>11.6f}")

    # Lloyd-Max（代表的なもの）
    for r in results["lloyd_max"]:
        if r["bits"] in [1, 2, 4, 8]:
            mse_str = f"{r['reconstruction_mse']:>11.6f}"
            print(f"{'Lloyd-Max':<12} {r['config']:<14} {r['memory_bits']:>8} b  {r['ip_error']:>11.6f}  {mse_str}")

    # QJL（代表的なもの）
    for r in results["qjl"]:
        if r["m"] in [100, 500, 1000, 2000]:
            print(f"{'QJL':<12} {r['config']:<14} {r['memory_bits']:>8} b  {r['ip_error_mean']:>11.6f}  {'N/A':>11}")

    # Multi-bit RP（代表的なもの）
    for r in results["multibit_rp"]:
        if (r["m"], r["b"]) in [(50, 2), (100, 2), (50, 4), (100, 4), (200, 4)]:
            print(f"{'Multi-bitRP':<12} {r['config']:<14} {r['memory_bits']:>8} b  {r['ip_error_mean']:>11.6f}  {'N/A':>11}")

    # TurboQuant
    for r in results.get("turbo_quant", []):
        print(f"{'TurboQuant':<12} {r['config']:<14} {r['memory_bits']:>8} b  {r['ip_error_mean']:>11.6f}  {'N/A':>11}")

    print()
    print(f"  True inner product = {true_ip:+.6f}")
