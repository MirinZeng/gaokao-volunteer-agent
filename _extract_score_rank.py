"""从两个一分一段表 PDF 抽取全部分数段，校验后存成 CSV。"""

import csv
import sys
from pathlib import Path

import pdfplumber

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(r"D:\Gaokaozhiyuan_Agent")
TARGET = PROJECT / "data" / "score_rank"
TARGET.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("物理", "江西省2025年普通高考物理类一分一段表.pdf"),
    ("历史", "江西省2025年普通高考历史类一分一段表.pdf"),
]


def parse_score(text: str) -> tuple[int, bool]:
    """'676及以上' -> (676, True)；'675' -> (675, False)。"""

    cleaned = text.strip().replace(" ", "")
    if cleaned.endswith("及以上"):
        return int(cleaned[: -len("及以上")]), True
    return int(cleaned), False


for subject, filename in SOURCES:
    path = PROJECT / filename
    rows: list[tuple[int, int, int, bool]] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for raw in table:
                    cells = ["" if cell is None else str(cell).strip().replace("\n", "") for cell in raw]
                    if len(cells) < 3 or cells[0] in {"分数", ""}:
                        continue
                    try:
                        score, is_top = parse_score(cells[0])
                        people = int(cells[1])
                        cumulative = int(cells[2])
                    except ValueError:
                        continue
                    rows.append((score, people, cumulative, is_top))

    print("=" * 70)
    print(f"{subject}类：共抽取 {len(rows)} 个分数段")
    print("=" * 70)

    # ---- 完整性校验 ----
    errors: list[str] = []
    scores = [row[0] for row in rows]
    if len(set(scores)) != len(scores):
        errors.append("分数有重复")
    for index in range(1, len(scores)):
        if scores[index] != scores[index - 1] - 1:
            errors.append(f"分数不连续：{scores[index - 1]} -> {scores[index]}")
    for index in range(1, len(rows)):
        expected = rows[index - 1][2] + rows[index][1]
        if rows[index][2] != expected:
            errors.append(
                f"累计人数不吻合：{scores[index]} 累计={rows[index][2]} 期望={expected}"
            )
    if rows[0][1] != rows[0][2]:
        errors.append(f"最高分段 人数({rows[0][1]}) != 累计({rows[0][2]})")

    print(f"  分数区间   : {scores[0]}（及以上） ~ {scores[-1]}")
    print(f"  总人数     : {rows[-1][2]:,}")
    print(f"  分段数     : {len(rows)}")
    print(f"  分数连续性 : {'✅ 连续无缺' if not any('不连续' in e for e in errors) else '❌ 有缺口'}")
    print(f"  累计一致性 : {'✅ 逐段吻合' if not any('不吻合' in e for e in errors) else '❌ 有偏差'}")
    if errors:
        print("  校验问题：")
        for item in errors[:10]:
            print(f"     - {item}")
    else:
        print("  ✅ 全部校验通过")

    out = TARGET / f"{subject}.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["分数", "人数", "累计人数", "是否含以上"])
        for score, people, cumulative, is_top in rows:
            writer.writerow([score, people, cumulative, int(is_top)])
    print(f"  已写入     : {out}")
    print()
