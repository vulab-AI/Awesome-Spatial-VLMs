#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 README / main.md 中解析 '## 🚀 Awesome Papers' 区域，
拿到每个 paper 的所有信息（层级 + 标签 + 标题 + 单位 + 链接），
结合上一次生成的 papers.json 判断新旧，并按添加时间从新到旧排序。
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

# ========= 配置区域（按需修改） =========

# 这里写你的 README / main.md 路径
# 如果你用的是 Awesome-Papers/main.md，就改成：
# README_PATH = Path("Awesome-Papers/main.md")
README_PATH = Path(os.getenv("PAPER_README_PATH", "../README.md"))

# 要解析的 section 标题（包含 🚀）
SECTION_HEADER = os.getenv("PAPER_SECTION_HEADER", "## 🚀 Awesome Papers")

# 输出 JSON 文件（用它作为“昨天的状态”来对比）
OUTPUT_JSON = Path(os.getenv("PAPER_OUTPUT_JSON", "./papers.json"))

# ======================================


def extract_section(text: str, header: str) -> str:
    """
    从全文中抽出某个二级标题（例如 '🚀 Awesome Papers'）下的内容，
    直到遇到下一个以 '#' 开头的标题（不含该标题）。
    """
    lines = text.splitlines()
    in_section = False
    collected = []
    for line in lines:
        print(f"Line: {line}", header in line.strip())
        if not in_section:
            if  header in line.strip() :
                in_section = True
            continue

        # 遇到下一个标题（#、##、### 等），说明这个 section 结束
        if line[:3] == "## ":
            break

        collected.append(line)

    return "\n".join(collected)



def parse_papers_with_hierarchy(section_text: str):
    """
    按行扫描：
    - 遇到 `### xxx` 记录当前 level3 分类
    - 遇到 `#### yyy` 记录当前 level4 子类
    - 遇到 bullet line：
        - [CoRR2023] Title (_Aff_) [[paper]](url) [[code]](...);
        - [NeurIPS2024] Title (Aff) [[paper]](url) ...
        - [COLING2025]; Title (Aff) [[paper]](url) ...
      解析出 label / title / affiliation / url / 分类
    """
    papers = []

    current_lv3 = ""  # 比如：Training-Free Prompting
    current_lv4 = ""  # 比如：Textual Prompting Methods

    # 匹配三级、四级标题
    heading_pattern = re.compile(r"^(#{3,4})\s+(.*)$")

    # label 部分：- [CoRR2023] xxx
    label_pattern = re.compile(r"^\s*[-*]\s+\[([^\]]+)\]\s*(.*)$")

    # 找第一个 [[paper]](...) 链接
    paper_link_pattern = re.compile(r"\[\[paper\]\]\(([^)]+)\)")
    # 兜底：如果有 [paper](...) 也尝试匹配
    paper_link_pattern_alt = re.compile(r"\[paper\]\(([^)]+)\)")

    for line in section_text.splitlines():
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            continue

        # 跳过注释
        if stripped.startswith("<!--"):
            continue

        # 跳过一长串横线 --------
        if set(stripped) == {"-"} and len(stripped) >= 3:
            continue

        # 是否是三级/四级标题
        mh = heading_pattern.match(stripped)
        if mh:
            hashes, title = mh.groups()
            level = len(hashes)
            title = title.strip()
            if level == 3:
                current_lv3 = title
            elif level == 4:
                current_lv4 = title
            continue

        # 不是标题，看是不是 bullet
        if not stripped.lstrip().startswith(("-", "*")):
            continue

        # 解析 label + 其余部分
        ml = label_pattern.match(raw)
        if not ml:
            continue

        label, rest = ml.groups()
        label = label.strip()
        rest = rest.strip()

        # 有些行是 "[arXiv2025]; Title ..."，去掉开头多余的分号和空格
        rest = rest.lstrip(" ;")

        # 找 [[paper]] 链接
        mp = paper_link_pattern.search(rest)
        url = None
        if mp:
            url = mp.group(1).strip()
        else:
            # 兜底：[paper](...)
            mp_alt = paper_link_pattern_alt.search(rest)
            if mp_alt:
                url = mp_alt.group(1).strip()

        if not url:
            # 没有 paper 链接就先跳过（你也可以改成继续收集）
            continue

        # 截取 [[paper]] 之前的部分，用来拆 title / affiliation
        before = rest[: (mp.start() if mp else mp_alt.start())].rstrip(" ;")

        # 尝试用最后一对括号当 affiliation
        lp = before.rfind("(")
        rp = before.rfind(")")
        if lp != -1 and rp != -1 and lp < rp:
            title = before[:lp].strip()
            affiliation = before[lp + 1 : rp].strip()
        else:
            title = before.strip()
            affiliation = ""

        # 去掉 affiliation 中的下划线、首尾空格
        affiliation = affiliation.replace("_", "").strip()

        papers.append(
            {
                "label": label,              # 如 CoRR2023 / CVPR2024
                "title": title,              # 论文标题
                "affiliation": affiliation,  # (_..._) 或 (...) 里的内容
                "url": url,                  # [[paper]] 链接
                "category_lv3": current_lv3,
                "category_lv4": current_lv4,
            }
        )

    return papers


def load_previous_state():
    """
    读取上一次生成的 papers.json（如果存在），返回：
    url -> {first_seen, last_seen, ...}
    """
    if not OUTPUT_JSON.exists():
        return {}

    with OUTPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prev_map = {}
    for p in data.get("papers", []):
        url = p.get("url")
        if not url:
            continue
        prev_map[url] = {
            "first_seen": p.get("first_seen"),
            "last_seen": p.get("last_seen"),
        }
    return prev_map


def main():
    if not README_PATH.exists():
        raise FileNotFoundError(f"{README_PATH} not found")

    text = README_PATH.read_text(encoding="utf-8")
    section_text = extract_section(text, SECTION_HEADER)
    if not section_text.strip():
        raise RuntimeError(f"Section '{SECTION_HEADER}' not found or empty in {README_PATH}")

    today_utc = datetime.now(timezone.utc).date().isoformat()  # "2025-12-11"

    print(f"Parsing section {SECTION_HEADER!r} from {README_PATH}")
    current_papers = parse_papers_with_hierarchy(section_text)
    print(f"Found {len(current_papers)} papers in current README.")

    # 读取“昨天”的状态（上一轮生成的 JSON）
    prev_state = load_previous_state()

    paper_entries = []
    new_papers_count = 0

    for p in current_papers:
        url = p["url"]

        if url in prev_state and prev_state[url].get("first_seen"):
            first_seen = prev_state[url]["first_seen"]
            is_new_today = False
        else:
            # 之前 JSON 中没有这个 URL -> 新论文
            first_seen = today_utc
            is_new_today = True
            new_papers_count += 1

        entry = {
            "label": p["label"],
            "title": p["title"],
            "affiliation": p["affiliation"],
            "url": p["url"],
            "category_lv3": p["category_lv3"],
            "category_lv4": p["category_lv4"],
            "first_seen": first_seen,        # 第一次在 README 出现的日期
            "last_seen": today_utc,          # 最近一次在 README 出现的日期
            "is_new_today": is_new_today,    # 本次运行中新加入的
        }
        paper_entries.append(entry)

    # 按 first_seen 从新到旧排序，新论文排前面
    paper_entries.sort(
        key=lambda x: (x["first_seen"], x["title"]),
        reverse=True,
    )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(README_PATH),
        "section_header": SECTION_HEADER,
        "count": len(paper_entries),
        "new_papers_today": new_papers_count,
        "papers": paper_entries,
    }

    OUTPUT_JSON.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"New papers today: {new_papers_count}")
    print(f"Wrote sorted JSON to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()