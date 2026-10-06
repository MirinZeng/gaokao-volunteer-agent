"""探查一分一段表 PDF 的结构：页数、文本、表格形态。"""

import sys
from pathlib import Path

import pdfplumber

sys.stdout.reconfigure(encoding="utf-8")

for name in ["江西省2025年普通高考物理类一分一段表.pdf", "江西省2025年普通高考历史类一分一段表.pdf"]:
    path = Path(name)
    print("=" * 76)
    print(f"文件：{name}")
    print("=" * 76)
    with pdfplumber.open(path) as pdf:
        print(f"页数：{len(pdf.pages)}")
        page = pdf.pages[0]
        print(f"页面尺寸：{page.width:.0f} x {page.height:.0f}")

        text = page.extract_text() or ""
        print("\n--- 第 1 页纯文本（前 700 字）---")
        print(text[:700])

        tables = page.extract_tables()
        print(f"\n--- 表格数量：{len(tables)} ---")
        for index, table in enumerate(tables[:2]):
            print(f"\n  表 {index + 1}：{len(table)} 行 x {len(table[0]) if table else 0} 列")
            for row_index, row in enumerate(table[:8]):
                cells = ["" if cell is None else str(cell).replace("\n", "⏎") for cell in row]
                print(f"    {row_index}: {cells}")
            print("    ...")
            for row_index, row in enumerate(table[-3:], start=len(table) - 3):
                cells = ["" if cell is None else str(cell).replace("\n", "⏎") for cell in row]
                print(f"    {row_index}: {cells}")
    print()
