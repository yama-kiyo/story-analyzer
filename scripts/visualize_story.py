#!/usr/bin/env python3
"""
ストーリーマップ可視化スクリプト
ストーリー分析JSONからダーク背景の感情アークグラフ＋スコアカードを生成する。

使い方:
    python visualize_story.py --config input.json --output story_map.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import numpy as np

# ---------------------------------------------------------------------------
# フォント検索
# ---------------------------------------------------------------------------
FONT_SEARCH_PATHS = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/Library/Fonts/ipag.ttf",
    os.path.expanduser("~/Library/Fonts/ipag.ttf"),
]


def find_font() -> str:
    for p in FONT_SEARCH_PATHS:
        if os.path.isfile(p):
            return p
    print("エラー: IPA Gothicフォント（ipag.ttf）が見つかりません。", file=sys.stderr)
    print("以下のいずれかのパスにフォントを配置してください:", file=sys.stderr)
    for p in FONT_SEARCH_PATHS:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# JSON バリデーション
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ["title", "arc_type_ja", "arc_ideal", "acts", "beats", "scores"]


def validate_config(data: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        print(f"エラー: JSONに必須キーがありません: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data["beats"], list) or len(data["beats"]) == 0:
        print("エラー: beatsは1つ以上の要素を持つ配列である必要があります。", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# スプライン補間（scipy があれば使う、なければ線形補間）
# ---------------------------------------------------------------------------
def smooth_curve(x, y, num=300):
    xs = np.linspace(min(x), max(x), num)
    try:
        from scipy.interpolate import make_interp_spline
        spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
        return xs, spline(xs)
    except ImportError:
        return xs, np.interp(xs, x, y)


# ---------------------------------------------------------------------------
# 色ユーティリティ
# ---------------------------------------------------------------------------
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4e"

ACT_COLORS = {
    1: (0.2, 0.4, 0.8, 0.08),   # 薄青
    "2a": (0.2, 0.7, 0.4, 0.08),  # 薄緑
    "2b": (0.8, 0.5, 0.2, 0.08),  # 薄橙
    3: (0.6, 0.3, 0.7, 0.08),   # 薄紫
}

GRADE_COLORS = {
    "A": "#4caf50",
    "B": "#2196f3",
    "C": "#ffc107",
    "D": "#ff9800",
    "F": "#f44336",
}

STATUS_MARKER = {
    "present": ("o", "#4caf50"),   # 緑丸
    "weak":    ("^", "#ffc107"),    # 黄三角
    "missing": ("X", "#f44336"),   # 赤バツ
}

SCORE_LABELS_JA = {
    "structure": "構造",
    "arc_clarity": "アーク明度",
    "pacing": "ペーシング",
    "engagement": "引き",
}


# ---------------------------------------------------------------------------
# メイン描画
# ---------------------------------------------------------------------------
def draw_story_map(data: dict, output_path: str, font_path: str) -> None:
    fp = FontProperties(fname=font_path)
    fp_small = FontProperties(fname=font_path, size=7)
    fp_mid = FontProperties(fname=font_path, size=9)
    fp_title = FontProperties(fname=font_path, size=14, weight="bold")
    fp_act = FontProperties(fname=font_path, size=11)
    fp_score = FontProperties(fname=font_path, size=10)

    # 1920x1080 @ 150dpi → figsize
    fig = plt.figure(figsize=(1920 / 150, 1080 / 150), dpi=150, facecolor=BG_COLOR)

    # 上段: 感情アーク（70%）、下段: スコアカード（30%）
    gs = fig.add_gridspec(2, 1, height_ratios=[7, 3], hspace=0.25,
                          left=0.06, right=0.94, top=0.93, bottom=0.06)
    ax_arc = fig.add_subplot(gs[0])
    ax_score = fig.add_subplot(gs[1])

    # --- 共通スタイル ---
    for ax in (ax_arc, ax_score):
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    # ===================================================================
    # 上段: 感情アークグラフ
    # ===================================================================
    acts = data["acts"]
    act1_end = acts.get("act1_end", 25)
    act2a_end = acts.get("act2a_end", 50)
    act2b_end = acts.get("act2b_end", 75)

    # 幕ごとの背景色
    ax_arc.axvspan(0, act1_end, color=ACT_COLORS[1])
    ax_arc.axvspan(act1_end, act2a_end, color=ACT_COLORS["2a"])
    ax_arc.axvspan(act2a_end, act2b_end, color=ACT_COLORS["2b"])
    ax_arc.axvspan(act2b_end, 100, color=ACT_COLORS[3])

    # 区切り破線
    for xpos in [act1_end, act2a_end, act2b_end]:
        ax_arc.axvline(xpos, color="white", linestyle="--", linewidth=0.7, alpha=0.4)

    # 幕ラベル（上部）
    act_labels = [
        ((0 + act1_end) / 2, "第1幕"),
        ((act1_end + act2a_end) / 2, "第2幕A"),
        ((act2a_end + act2b_end) / 2, "第2幕B"),
        ((act2b_end + 100) / 2, "第3幕"),
    ]
    for xc, label in act_labels:
        ax_arc.text(xc, 1.12, label, ha="center", va="bottom", color=TEXT_COLOR,
                    fontproperties=fp_act, alpha=0.7)

    # 起承転結ラベル（X軸下）
    kstk = data.get("kishotenketsu", {"ki": "起", "sho": "承", "ten": "転", "ketsu": "結"})
    kstk_labels = [
        ((0 + act1_end) / 2, kstk.get("ki", "起")),
        ((act1_end + act2a_end) / 2, kstk.get("sho", "承")),
        ((act2a_end + act2b_end) / 2, kstk.get("ten", "転")),
        ((act2b_end + 100) / 2, kstk.get("ketsu", "結")),
    ]
    for xc, label in kstk_labels:
        ax_arc.text(xc, -1.22, label, ha="center", va="top", color=TEXT_COLOR,
                    fontproperties=fp_act, alpha=0.6)

    # 理想アーク（帯）
    ideal = data["arc_ideal"]
    ix = [pt[0] for pt in ideal]
    iy = [pt[1] for pt in ideal]
    ixs, iys = smooth_curve(ix, iy)
    ax_arc.fill_between(ixs, iys - 0.08, iys + 0.08, color="#7c4dff", alpha=0.15)
    ax_arc.plot(ixs, iys, color="#7c4dff", linewidth=1.5, alpha=0.3, linestyle="--")

    # ビート曲線
    beats = sorted(data["beats"], key=lambda b: b["position"])
    bx = [b["position"] for b in beats]
    by = [b["valence"] for b in beats]
    bxs, bys = smooth_curve(bx, by)
    ax_arc.plot(bxs, bys, color="white", linewidth=2.0, alpha=0.85, zorder=3)

    # ビートマーカー
    for b in beats:
        status = b.get("status", "present")
        marker, color = STATUS_MARKER.get(status, ("o", "#4caf50"))
        ax_arc.scatter(b["position"], b["valence"], marker=marker, color=color,
                       s=60, zorder=5, edgecolors="white", linewidths=0.5)

        # ラベル（name_ja + label）
        label_text = b.get("name_ja", b.get("name", ""))
        beat_label = b.get("label", "")
        if beat_label:
            label_text += f"\n{beat_label}"

        # ラベル位置の微調整（上下交互）
        offset_y = 0.10 if b["valence"] >= 0 else -0.10
        va = "bottom" if offset_y > 0 else "top"
        ax_arc.annotate(
            label_text,
            xy=(b["position"], b["valence"]),
            xytext=(0, 12 if offset_y > 0 else -12),
            textcoords="offset points",
            ha="center", va=va,
            fontproperties=fp_small,
            color=TEXT_COLOR, alpha=0.85,
            zorder=6,
        )

    # 軸設定
    ax_arc.set_xlim(-2, 102)
    ax_arc.set_ylim(-1.15, 1.15)
    ax_arc.set_xlabel("ストーリー進行 (%)", fontproperties=fp_mid, color=TEXT_COLOR, labelpad=18)
    ax_arc.set_ylabel("感情バレンス", fontproperties=fp_mid, color=TEXT_COLOR)
    ax_arc.axhline(0, color=GRID_COLOR, linewidth=0.5)
    ax_arc.set_xticks(range(0, 101, 10))
    ax_arc.set_xticklabels([f"{i}%" for i in range(0, 101, 10)], fontproperties=fp_small)
    ax_arc.tick_params(axis="y", labelsize=7)

    # タイトル（左上）とアーク型（右上）
    ax_arc.text(0.01, 1.02, data["title"], transform=ax_arc.transAxes,
                fontproperties=fp_title, color="white", va="bottom")
    ax_arc.text(0.99, 1.02, data.get("arc_type_ja", ""), transform=ax_arc.transAxes,
                fontproperties=fp_act, color="#b0b0b0", va="bottom", ha="right")

    # ===================================================================
    # 下段: スコアカード
    # ===================================================================
    scores = data["scores"]
    categories = list(scores.keys())
    values = [scores[k]["value"] for k in categories]
    grades = [scores[k]["grade"] for k in categories]
    colors = [GRADE_COLORS.get(g, "#888888") for g in grades]
    labels_ja = [SCORE_LABELS_JA.get(k, k) for k in categories]

    y_pos = np.arange(len(categories))
    bars = ax_score.barh(y_pos, values, height=0.55, color=colors, alpha=0.85, zorder=3)

    ax_score.set_yticks(y_pos)
    ax_score.set_yticklabels(labels_ja, fontproperties=fp_score)
    ax_score.set_xlim(0, 110)
    ax_score.set_xlabel("スコア", fontproperties=fp_mid, color=TEXT_COLOR)
    ax_score.invert_yaxis()
    ax_score.axvline(100, color=GRID_COLOR, linewidth=0.5, linestyle=":")

    # バーの上にグレードと数値
    for i, (bar, grade, val) in enumerate(zip(bars, grades, values)):
        ax_score.text(val + 1.5, i, f"{grade}  {val}", va="center", ha="left",
                      fontproperties=fp_score, color=TEXT_COLOR)

    # グリッド
    ax_score.set_axisbelow(True)
    ax_score.grid(axis="x", color=GRID_COLOR, linewidth=0.3, alpha=0.5)
    ax_score.tick_params(axis="x", labelsize=7)

    # --- 保存 ---
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"出力完了: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ストーリーマップ可視化")
    parser.add_argument("--config", required=True, help="入力JSONファイルのパス")
    parser.add_argument("--output", required=True, help="出力画像ファイルのパス (.png)")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"エラー: 入力ファイルが見つかりません: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"エラー: JSONの解析に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    validate_config(data)
    font_path = find_font()
    draw_story_map(data, args.output, font_path)


if __name__ == "__main__":
    main()
