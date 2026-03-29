"""
可視化モジュール

実験結果のグラフ生成とアニメーション作成を行う。

生成される出力:
  静的グラフ:
    1. QJL 収束（論文版、信頼区間付き）
    2. QJL 比較（論文版 vs 対称版 — バイアスの違いを示す）
    3. Lloyd-Max 量子化器の可視化
    4. Lloyd-Max MSE 収束
    5. ランダム回転の効果
    6. TurboQuant 2段階アプローチ
    7. メモリ vs 内積誤差の比較（全手法）
    8. サマリテーブル
  アニメーション:
    9.  Lloyd-Max 学習過程
    10. QJL 収束過程（論文版）
    11. TurboQuant 2段階収束
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .qjl import qjl_incremental, qjl_incremental_unbiased, qjl_convergence_unbiased
from .lloyd_max import lloyd_max_train
from .turbo_quant import random_rotation_matrix, turbo_quant_prod_incremental
from .utils import generate_samples, generate_vectors_with_ip


# ============================================================
# 1. QJL 収束グラフ
# ============================================================

def plot_qjl_convergence(results, save_path="outputs/figures/qjl_convergence.png"):
    """
    論文版 QJL（片側量子化）の収束を信頼区間付きで可視化する。
    """
    m_values = results["m_values"]
    true_ip = results["true_ip"]
    conv = results["qjl_conv_unbiased"]
    means = conv["means"]
    stds = conv["stds"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(m_values, means, "o-", color="steelblue",
            label="Paper QJL (unbiased, mean)", markersize=5, linewidth=1.5, zorder=3)
    ax.fill_between(m_values, means - stds, means + stds,
                    alpha=0.2, color="steelblue", label="$\\pm 1\\sigma$")
    ax.axhline(y=true_ip, color="crimson", linestyle="--", linewidth=1.5,
               label=f"True inner product = {true_ip:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of projections (m)", fontsize=12)
    ax.set_ylabel("Estimated inner product", fontsize=12)
    ax.set_title("QJL Inner Product Estimation (Paper Version)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_qjl_comparison(save_path="outputs/figures/qjl_comparison.png"):
    """
    論文版 QJL（厳密に不偏） vs 対称版 QJL（近似的に不偏）の比較。

    内積が大きいベクトル（⟨x,y⟩ = 0.5）を使い、
    対称版のバイアスが目に見える状況で比較する。
    """
    # 内積 0.5 のベクトルペアを生成（バイアスが見える設定）
    x, y, true_ip = generate_vectors_with_ip(d=100, target_ip=0.5, seed=99)
    m_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    # 両版の収束データ
    from .qjl import qjl_convergence
    conv_sym = qjl_convergence(x, y, m_values, n_trials=50, seed=200)
    conv_paper = qjl_convergence_unbiased(x, y, m_values, n_trials=50, seed=201)

    # 対称版の理論的収束先: π/2 - arccos(⟨x,y⟩)
    biased_target = np.pi / 2 - np.arccos(true_ip)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左: 論文版（厳密に不偏） ---
    means_p = conv_paper["means"]
    stds_p = conv_paper["stds"]
    ax1.plot(m_values, means_p, "o-", color="steelblue",
             label="Estimate (mean)", markersize=5, linewidth=1.5)
    ax1.fill_between(m_values, means_p - stds_p, means_p + stds_p,
                     alpha=0.2, color="steelblue")
    ax1.axhline(y=true_ip, color="crimson", linestyle="--", linewidth=1.5,
                label=f"True = {true_ip:.4f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Projections (m)", fontsize=11)
    ax1.set_ylabel("Estimated inner product", fontsize=11)
    ax1.set_title("Paper QJL (Exactly Unbiased)\n"
                  "$\\sqrt{\\pi/2}/m \\cdot \\Sigma\\, (s_k^T y) \\cdot \\mathrm{sign}(s_k^T x)$",
                  fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 0.7)

    # --- 右: 対称版（近似的に不偏） ---
    means_s = conv_sym["means"]
    stds_s = conv_sym["stds"]
    ax2.plot(m_values, means_s, "o-", color="darkorange",
             label="Estimate (mean)", markersize=5, linewidth=1.5)
    ax2.fill_between(m_values, means_s - stds_s, means_s + stds_s,
                     alpha=0.2, color="darkorange")
    ax2.axhline(y=true_ip, color="crimson", linestyle="--", linewidth=1.5,
                label=f"True = {true_ip:.4f}")
    ax2.axhline(y=biased_target, color="gray", linestyle=":", linewidth=1.5,
                label=f"Convergence target = {biased_target:.4f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Projections (m)", fontsize=11)
    ax2.set_title("Symmetric QJL (Approx. Unbiased)\n"
                  "$(\\pi/2) \\cdot (1/m) \\cdot \\Sigma\\, \\mathrm{sign}(s_k^T x) "
                  "\\cdot \\mathrm{sign}(s_k^T y)$",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 0.7)

    fig.suptitle(
        f"QJL Comparison: Paper vs Symmetric  "
        f"($\\langle x, y \\rangle$ = {true_ip:.3f})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_random_rotation_effect(d=100, seed=42,
                                save_path="outputs/figures/random_rotation_effect.png"):
    """
    ランダム回転がベクトル座標の分布に与える効果を可視化する。

    回転前: 座標は特定のベクトルに依存した非一様な分布
    回転後: 座標が ≈ i.i.d. N(0, 1/d) に従う
    → スカラー量子化が各座標で独立に適用可能になる
    """
    rng = np.random.default_rng(seed)

    # 偏った単位ベクトルを作成（一部の座標に集中）
    x = np.zeros(d)
    x[:10] = rng.standard_normal(10)   # 最初の10成分にエネルギーが集中
    x = x / np.linalg.norm(x)

    # ランダム回転
    Pi = random_rotation_matrix(d, seed=seed + 100)
    x_rotated = Pi @ x

    # 理論分布: N(0, 1/d)
    t = np.linspace(-0.5, 0.5, 200)
    sigma = 1 / np.sqrt(d)
    theoretical = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-t ** 2 / (2 * sigma ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 回転前
    ax1.bar(range(d), x, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Coordinate index", fontsize=11)
    ax1.set_ylabel("Value", fontsize=11)
    ax1.set_title("Before Rotation\n(energy concentrated in few coords)", fontsize=11)
    ax1.set_ylim(-0.5, 0.5)
    ax1.grid(True, alpha=0.3)

    # 回転後
    ax2.hist(x_rotated, bins=25, density=True, alpha=0.6, color="mediumseagreen",
             label="Rotated coordinates")
    ax2.plot(t, theoretical, "r-", linewidth=2,
             label=f"$N(0, 1/d)$, $d={d}$")
    ax2.set_xlabel("Value", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("After Random Rotation\n(≈ i.i.d. Gaussian)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Random Rotation: Making Coordinates i.i.d.",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_turbo_quant_two_stage(x, y, true_ip, d=100,
                               save_path="outputs/figures/turbo_quant_two_stage.png"):
    """
    TurboQuant の2段階アプローチを可視化する。

    左: 各 b での MSE のみ vs 2段階（バイアス補正の効果）
    右: QJL 射影数を増やした時の収束（MSE 開始点から真値へ）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左: 各ビット幅での比較 ---
    b_values = [2, 3, 4]
    bar_width = 0.25
    x_pos = np.arange(len(b_values))

    mse_only_ips = []
    two_stage_ips = []

    for b in b_values:
        ests_2stage, ip_mse = turbo_quant_prod_incremental(
            x, y, b, max_m=500, rotation_seed=42, lloyd_seed=42, qjl_seed=123
        )
        mse_only_ips.append(ip_mse)
        two_stage_ips.append(np.mean(ests_2stage[-50:]))  # 最後50本の平均

    ax1.bar(x_pos - bar_width / 2, mse_only_ips, bar_width,
            color="darkorange", alpha=0.8, label="MSE only (biased)")
    ax1.bar(x_pos + bar_width / 2, two_stage_ips, bar_width,
            color="steelblue", alpha=0.8, label="MSE + QJL (unbiased)")
    ax1.axhline(y=true_ip, color="crimson", linestyle="--", linewidth=1.5,
                label=f"True = {true_ip:.4f}")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"b={b}" for b in b_values])
    ax1.set_ylabel("Inner product estimate", fontsize=11)
    ax1.set_title("Two-Stage Correction Effect", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- 右: b=3 での収束過程 ---
    estimates_b3, ip_mse_b3 = turbo_quant_prod_incremental(
        x, y, b=3, max_m=500, rotation_seed=42, lloyd_seed=42, qjl_seed=456
    )

    m_range = range(1, 501)
    ax2.plot(m_range, estimates_b3, "-", color="steelblue", linewidth=1,
             alpha=0.7, label="TurboQuant (b=3)")
    ax2.axhline(y=true_ip, color="crimson", linestyle="--", linewidth=1.5,
                label=f"True = {true_ip:.4f}")
    ax2.axhline(y=ip_mse_b3, color="darkorange", linestyle=":", linewidth=1.5,
                label=f"MSE only = {ip_mse_b3:.4f}")
    ax2.set_xlabel("QJL projections on residual", fontsize=11)
    ax2.set_ylabel("Inner product estimate", fontsize=11)
    ax2.set_title("Two-Stage Convergence (b=3)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("TurboQuant: MSE Quantization + Residual QJL",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 2. Lloyd-Max 量子化器の可視化
# ============================================================

def plot_lloyd_max_quantizer(n_levels_list=None, n_samples=10000, seed=42,
                             save_path="outputs/figures/lloyd_max_quantizer.png"):
    """
    Lloyd-Max 量子化器をレベル数ごとに可視化する。

    ヒストグラム上に決定境界とセントロイドを重ねて表示。
    レベル数が増えるほど MSE が減少する。
    """
    if n_levels_list is None:
        n_levels_list = [2, 4, 8]

    samples = generate_samples(n=n_samples, seed=seed)

    n_plots = len(n_levels_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, n_levels in zip(axes, n_levels_list):
        centroids, boundaries, history = lloyd_max_train(samples, n_levels)
        final_mse = history[-1]["mse"]
        bits = int(np.log2(n_levels))

        # ヒストグラム
        ax.hist(samples, bins=80, density=True, alpha=0.5, color="gray",
                label="Data distribution")

        # 決定境界
        for i, b in enumerate(boundaries):
            label = "Boundaries" if i == 0 else None
            ax.axvline(x=b, color="crimson", linestyle="--",
                       linewidth=1.2, alpha=0.8, label=label)

        # セントロイド
        for i, c in enumerate(centroids):
            label = "Centroids" if i == 0 else None
            ax.plot(c, 0.02, "o", color="steelblue", markersize=10,
                    zorder=5, label=label)

        ax.set_title(f"{n_levels} levels ({bits} bit)\nMSE = {final_mse:.4f}",
                     fontsize=11)
        ax.set_xlabel("Value", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(-4, 4)

    axes[0].set_ylabel("Density", fontsize=10)
    fig.suptitle("Lloyd-Max Quantizer: Learned Centroids and Boundaries",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 3. Lloyd-Max MSE 収束
# ============================================================

def plot_lloyd_max_mse(n_levels=4, n_samples=10000, seed=42,
                       save_path="outputs/figures/lloyd_max_mse.png"):
    """
    Lloyd-Max 学習中の MSE 推移を可視化する。

    反復を重ねるごとに MSE が単調減少し収束する様子を確認できる。
    """
    samples = generate_samples(n=n_samples, seed=seed)
    _, _, history = lloyd_max_train(samples, n_levels, max_iter=30)

    iterations = [h["iteration"] + 1 for h in history]
    mses = [h["mse"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, mses, "o-", color="darkorange", markersize=6, linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("MSE (Reconstruction Error)", fontsize=12)
    ax.set_title(f"Lloyd-Max Training: MSE Convergence ({n_levels} levels)",
                 fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 4. メモリ vs 内積誤差の比較（最重要）
# ============================================================

def plot_comparison_memory_vs_error(results,
                                     save_path="outputs/figures/comparison_memory_vs_error.png"):
    """
    メモリ使用量 vs 内積誤差の比較グラフを全手法で作成する。

    これがプロジェクトの核心となるグラフ。
    同じメモリ予算でどの手法が最も内積を正確に推定できるかが一目で分かる。

    x軸: メモリ使用量（ビット/ベクトル、対数スケール）
    y軸: 内積推定誤差（|推定値 - 真値|）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Baseline ---
    bl = results["baseline"]
    ax.plot(bl["memory_bits"], bl["ip_error"], "*", color="black",
            markersize=18, label="Baseline (float32)", zorder=5)

    # --- Lloyd-Max ---
    lm_mem = [r["memory_bits"] for r in results["lloyd_max"]]
    lm_err = [r["ip_error"] for r in results["lloyd_max"]]
    ax.plot(lm_mem, lm_err, "s-", color="darkorange", markersize=8,
            linewidth=1.5, label="Lloyd-Max", zorder=4)

    # --- QJL ---
    qjl_mem = [r["memory_bits"] for r in results["qjl"]]
    qjl_err = [r["ip_error_mean"] for r in results["qjl"]]
    ax.plot(qjl_mem, qjl_err, "o-", color="steelblue", markersize=7,
            linewidth=1.5, label="QJL (1-bit proj.)", zorder=4)

    # --- Hybrid（ビット数ごとに分けてプロット）---
    hybrid_b_values = sorted(set(r["b"] for r in results["hybrid"]))
    hybrid_colors = {2: "mediumseagreen", 4: "mediumpurple"}
    hybrid_markers = {2: "^", 4: "v"}

    for b_val in hybrid_b_values:
        h_filtered = [r for r in results["hybrid"] if r["b"] == b_val]
        h_filtered.sort(key=lambda r: r["memory_bits"])
        h_mem = [r["memory_bits"] for r in h_filtered]
        h_err = [r["ip_error_mean"] for r in h_filtered]
        color = hybrid_colors.get(b_val, "gray")
        marker = hybrid_markers.get(b_val, "D")
        ax.plot(h_mem, h_err, f"{marker}-", color=color, markersize=7,
                linewidth=1.5, label=f"Hybrid ({b_val}-bit proj.)", zorder=4)

    # --- TurboQuant ---
    tq_data = results.get("turbo_quant", [])
    if tq_data:
        tq_b_values = sorted(set(r["b"] for r in tq_data))
        tq_colors = {2: "deeppink", 3: "brown", 4: "teal"}
        for b_val in tq_b_values:
            t_filtered = [r for r in tq_data if r["b"] == b_val]
            t_filtered.sort(key=lambda r: r["memory_bits"])
            t_mem = [r["memory_bits"] for r in t_filtered]
            t_err = [r["ip_error_mean"] for r in t_filtered]
            color = tq_colors.get(b_val, "gray")
            ax.plot(t_mem, t_err, "D-", color=color, markersize=7,
                    linewidth=1.5, label=f"TurboQuant (b={b_val})", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Memory (bits per vector)", fontsize=12)
    ax.set_ylabel("Inner Product Error  $|\\langle x,y \\rangle - \\hat{v}|$",
                  fontsize=12)
    ax.set_title("Memory vs Inner Product Accuracy: All Methods", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 5. サマリテーブル（図として保存）
# ============================================================

def plot_summary_table(results, save_path="outputs/figures/summary_table.png"):
    """
    全手法の比較結果を表形式のグラフとして保存する。
    """
    # 表示する行データを収集
    rows = []

    # Baseline
    bl = results["baseline"]
    rows.append(["Baseline", "float32", f"{bl['memory_bits']}", "0.000000", "0.000000"])

    # Lloyd-Max（代表的な設定）
    for r in results["lloyd_max"]:
        if r["bits"] in [1, 2, 4, 8]:
            rows.append([
                "Lloyd-Max", r["config"], f"{r['memory_bits']}",
                f"{r['ip_error']:.6f}", f"{r['reconstruction_mse']:.6f}",
            ])

    # QJL（代表的な設定）
    for r in results["qjl"]:
        if r["m"] in [100, 500, 1000, 2000]:
            rows.append([
                "QJL", r["config"], f"{r['memory_bits']}",
                f"{r['ip_error_mean']:.6f}", "N/A",
            ])

    # Hybrid（代表的な設定）
    for r in results["hybrid"]:
        if (r["m"], r["b"]) in [(50, 2), (100, 2), (50, 4), (100, 4), (200, 4)]:
            rows.append([
                "Hybrid", r["config"], f"{r['memory_bits']}",
                f"{r['ip_error_mean']:.6f}", "N/A",
            ])

    # TurboQuant
    for r in results.get("turbo_quant", []):
        rows.append([
            "TurboQuant", r["config"], f"{r['memory_bits']}",
            f"{r['ip_error_mean']:.6f}", "N/A",
        ])

    # テーブル描画
    col_labels = ["Method", "Config", "Memory (bits)", "IP Error", "Recon MSE"]
    n_rows = len(rows)

    n_cols = len(col_labels)
    # 行の高さの比率: タイトル行 + ヘッダ行 + データ行
    total_rows = n_rows + 2  # +1 title, +1 header
    row_h = 1.0 / total_rows

    fig, ax = plt.subplots(figsize=(10, 0.32 * total_rows))
    ax.axis("off")

    # タイトルバー（表の上に直結する矩形 + テキスト）
    gap = row_h * 0.35  # タイトルと表の間の隙間
    title_y = 1.0 - row_h
    ax.add_patch(plt.Rectangle(
        (0, title_y), 1.0, row_h,
        transform=ax.transAxes, facecolor="white",
        edgecolor="none", clip_on=False,
    ))
    ax.text(0.5, title_y + row_h / 2,
            "Quantization Methods: Comparison Summary",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=13, fontweight="bold", color="black")

    # テーブル本体（タイトルバーの下に配置、gap 分だけ下げる）
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, title_y - gap],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # ヘッダ行の色
    for j in range(n_cols):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # データ行の色分け（手法ごと）
    method_colors = {
        "Baseline": "#F2F2F2",
        "Lloyd-Max": "#FFF2CC",
        "QJL": "#D6E4F0",
        "Hybrid": "#E2EFDA",
        "TurboQuant": "#FCE4EC",
    }
    for i, row in enumerate(rows):
        color = method_colors.get(row[0], "white")
        for j in range(n_cols):
            table[i + 1, j].set_facecolor(color)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 6. Lloyd-Max 学習アニメーション
# ============================================================

def animate_lloyd_max(n_levels=4, n_samples=10000, seed=42,
                      save_path="outputs/animations/lloyd_max_training.gif"):
    """
    Lloyd-Max 量子化器の学習過程をアニメーションにする。

    各フレームで境界とセントロイドが最適位置に向かって
    移動する様子を表示する。
    """
    samples = generate_samples(n=n_samples, seed=seed)
    _, _, history = lloyd_max_train(samples, n_levels, max_iter=30)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ヒストグラムは固定背景
    ax.hist(samples, bins=80, density=True, alpha=0.4, color="gray")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)

    dynamic_artists = []

    def update(frame):
        for artist in dynamic_artists:
            artist.remove()
        dynamic_artists.clear()

        h = history[frame]

        for b in h["boundaries"]:
            line = ax.axvline(x=b, color="crimson", linestyle="--",
                              linewidth=1.5, alpha=0.8)
            dynamic_artists.append(line)

        for c in h["centroids"]:
            marker, = ax.plot(c, 0.02, "o", color="steelblue",
                              markersize=12, zorder=5)
            dynamic_artists.append(marker)

        ax.set_title(
            f"Lloyd-Max Training  |  Iteration {h['iteration'] + 1}  |  "
            f"MSE = {h['mse']:.6f}",
            fontsize=12,
        )

    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=500, repeat=True)
    anim.save(save_path, writer=PillowWriter(fps=2))
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 7. QJL 収束アニメーション
# ============================================================

def animate_qjl(x, y, true_ip, max_m=500, seed=123,
                save_path="outputs/animations/qjl_convergence.gif"):
    """
    論文版 QJL（片側量子化）の収束過程をアニメーションにする。

    射影を1本ずつ追加し、厳密に不偏な推定値が
    真の内積に近づいていく軌跡を表示する。
    """
    estimates = qjl_incremental_unbiased(x, y, max_m=max_m, seed=seed)

    # フレームのサブサンプリング（最初は細かく、後は粗く）
    frame_indices = []
    frame_indices.extend(range(0, min(20, max_m)))
    frame_indices.extend(range(20, min(100, max_m), 5))
    frame_indices.extend(range(100, min(300, max_m), 20))
    frame_indices.extend(range(300, max_m, 50))
    frame_indices = sorted(set(frame_indices))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(y=true_ip, color="crimson", linestyle="--", linewidth=1.5,
               label=f"True inner product = {true_ip:.4f}")

    line, = ax.plot([], [], "-", color="steelblue", linewidth=1, alpha=0.7)
    point, = ax.plot([], [], "o", color="steelblue", markersize=8, zorder=5)

    ax.set_xlim(1, max_m * 1.05)
    y_range = max(abs(true_ip) + 1.5, 2.0)
    ax.set_ylim(true_ip - y_range, true_ip + y_range)
    ax.set_xlabel("Number of projections (m)", fontsize=11)
    ax.set_ylabel("Estimated inner product", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    def update(frame_num):
        idx = frame_indices[frame_num]
        m_range = list(range(1, idx + 2))
        est_range = estimates[:idx + 1]

        line.set_data(m_range, est_range)
        point.set_data([m_range[-1]], [est_range[-1]])
        ax.set_title(
            f"QJL Convergence  |  m = {idx + 1}  |  "
            f"estimate = {est_range[-1]:.4f}",
            fontsize=12,
        )

        return line, point

    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                         interval=100, repeat=True)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 8. TurboQuant 2段階収束アニメーション
# ============================================================

def animate_turbo_quant(x, y, true_ip, b=3, max_m=500,
                        rotation_seed=42, lloyd_seed=42, qjl_seed=456,
                        save_path="outputs/animations/turbo_quant_convergence.gif"):
    """
    TurboQuant 2段階アプローチの収束アニメーション。

    MSE量子化のみの（バイアス付き）推定値を開始点として、
    残差への QJL 射影を1本ずつ追加し、推定値が真の内積に
    収束していく過程を表示する。
    """
    estimates, ip_mse = turbo_quant_prod_incremental(
        x, y, b, max_m=max_m,
        rotation_seed=rotation_seed, lloyd_seed=lloyd_seed, qjl_seed=qjl_seed,
    )

    # フレームのサブサンプリング
    frame_indices = []
    frame_indices.extend(range(0, min(20, max_m)))
    frame_indices.extend(range(20, min(100, max_m), 5))
    frame_indices.extend(range(100, min(300, max_m), 20))
    frame_indices.extend(range(300, max_m, 50))
    frame_indices = sorted(set(frame_indices))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(y=true_ip, color="crimson", linestyle="--", linewidth=1.5,
               label=f"True inner product = {true_ip:.4f}")
    ax.axhline(y=ip_mse, color="darkorange", linestyle=":", linewidth=1.5,
               label=f"MSE only (biased) = {ip_mse:.4f}")

    line, = ax.plot([], [], "-", color="steelblue", linewidth=1, alpha=0.7)
    point, = ax.plot([], [], "o", color="steelblue", markersize=8, zorder=5)

    ax.set_xlim(1, max_m * 1.05)
    margin = max(abs(true_ip - ip_mse) * 3, 0.5)
    center = (true_ip + ip_mse) / 2
    ax.set_ylim(center - margin, center + margin)
    ax.set_xlabel("QJL projections on residual", fontsize=11)
    ax.set_ylabel("Inner product estimate", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    def update(frame_num):
        idx = frame_indices[frame_num]
        m_range = list(range(1, idx + 2))
        est_range = estimates[:idx + 1]

        line.set_data(m_range, est_range)
        point.set_data([m_range[-1]], [est_range[-1]])
        ax.set_title(
            f"TurboQuant (b={b})  |  QJL projections = {idx + 1}  |  "
            f"estimate = {est_range[-1]:.4f}",
            fontsize=12,
        )

        return line, point

    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                         interval=100, repeat=True)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close()
    print(f"  Saved: {save_path}")
