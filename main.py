"""
QJL vs MSE最小化量子化 — 5手法比較プロジェクト

5つの量子化手法を「内積精度」「再構成誤差」「メモリ使用量」の
3軸で比較し、量子化におけるトレードオフを可視化する。

比較対象:
  1. Baseline    — float32 のまま内積を計算（基準）
  2. Lloyd-Max   — MSE最小化スカラー量子化 → 再構成は良いが内積にバイアス
  3. QJL         — ランダム射影 + 1ビット量子化 → 内積を不偏推定
  4. Hybrid      — ランダム射影 + 低ビット量子化 → メモリと精度のバランス
  5. TurboQuant  — MSE量子化 + 残差QJL → 再構成と内積の両立

実行すると以下を生成する:
  outputs/figures/     比較グラフ（PNG）
  outputs/animations/  学習・収束過程のアニメーション（GIF）
"""

import os

import matplotlib
matplotlib.use("Agg")

from src.utils import generate_vectors
from src.experiments import run_all_experiments, print_summary_table
from src.visualization import (
    plot_qjl_convergence,
    plot_qjl_comparison,
    plot_random_rotation_effect,
    plot_turbo_quant_two_stage,
    plot_lloyd_max_quantizer,
    plot_lloyd_max_mse,
    plot_comparison_memory_vs_error,
    plot_summary_table,
    animate_lloyd_max,
    animate_qjl,
    animate_turbo_quant,
)


def main():
    # ============================================================
    # 設定
    # ============================================================
    d = 100         # ベクトルの次元数
    seed = 42       # 乱数シード

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/animations", exist_ok=True)

    # ベクトル生成
    x, y, true_ip = generate_vectors(d=d, seed=seed)

    print("=" * 60)
    print("  5-Method Quantization Comparison")
    print("  Baseline / Lloyd-Max / QJL / Hybrid / TurboQuant")
    print("=" * 60)
    print(f"  Dimension       d = {d}")
    print(f"  True inner product = {true_ip:.6f}")
    print()

    # ============================================================
    # 実験の実行
    # ============================================================
    print("[1/8] Running all experiments...")
    results = run_all_experiments(x, y, d, seed=seed)

    # サマリ表示
    print_summary_table(results)

    # ============================================================
    # 可視化
    # ============================================================

    # ① QJL 収束（論文版）
    print("[2/8] QJL convergence plot (paper version)...")
    plot_qjl_convergence(results)

    # ② QJL 比較（論文版 vs 対称版）
    print("[3/8] QJL comparison (paper vs symmetric)...")
    plot_qjl_comparison()

    # ③ Lloyd-Max 量子化器
    print("[4/8] Lloyd-Max quantizer & MSE plots...")
    plot_lloyd_max_quantizer(n_levels_list=[2, 4, 8])
    plot_lloyd_max_mse(n_levels=4)

    # ④ ランダム回転の効果
    print("[5/8] Random rotation effect...")
    plot_random_rotation_effect(d=d, seed=seed)

    # ⑤ TurboQuant 2段階アプローチ
    print("[6/8] TurboQuant two-stage visualization...")
    plot_turbo_quant_two_stage(x, y, true_ip, d=d)

    # ⑥ メモリ vs 内積誤差（最重要）
    print("[7/8] Memory vs IP error comparison (key figure)...")
    plot_comparison_memory_vs_error(results)

    # ⑦ サマリテーブル
    plot_summary_table(results)

    # ============================================================
    # アニメーション
    # ============================================================
    print("[8/8] Generating animations...")
    animate_lloyd_max(n_levels=4)
    animate_qjl(x, y, true_ip, max_m=500, seed=seed + 10)
    animate_turbo_quant(x, y, true_ip, b=3, max_m=500, qjl_seed=seed + 20)

    # ============================================================
    # 完了
    # ============================================================
    print()
    print("=" * 60)
    print("  Done!")
    print("=" * 60)
    print("  Figures:    outputs/figures/")
    print("  Animations: outputs/animations/")
    print()


if __name__ == "__main__":
    main()
