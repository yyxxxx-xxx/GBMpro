#!/usr/bin/env python3
"""
生成 Llamole-main 项目的中文思维导图（基于 gbm_project/RESEARCH_METHODOLOGY.md）
脚本功能：
 - 解析 Markdown 标题层级
 - 生成 Graphviz dot 文件
 - 尝试下载中文字体（Source Han Sans SC），刷新字体缓存
 - 使用 dot 渲染为 PNG

注意：如果运行环境缺少 graphviz 的系统二进制（dot），脚本会仅生成 dot 文件供离线渲染。
"""
import os
import re
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD_PATH = os.path.join(ROOT, "gbm_project", "RESEARCH_METHODOLOGY.md")
OUT_DIR = os.path.join(ROOT, "experiments", "mindmaps")
os.makedirs(OUT_DIR, exist_ok=True)
DOT_PATH = os.path.join(OUT_DIR, "llamole_gbm_mindmap.dot")
PNG_PATH = os.path.join(OUT_DIR, "llamole_gbm_mindmap.png")

FONT_DIR = os.path.expanduser("~/.local/share/fonts")
os.makedirs(FONT_DIR, exist_ok=True)
FONT_PATH = os.path.join(FONT_DIR, "SourceHanSansSC-Regular.otf")
FONT_URLS = [
    "https://github.com/adobe-fonts/source-han-sans/releases/download/2.040R/SourceHanSansSC-Regular.otf",
    "https://github.com/adobe-fonts/source-han-sans/raw/release/SourceHanSansSC-Regular.otf"
]

def try_install_font():
    if os.path.exists(FONT_PATH) and os.path.getsize(FONT_PATH) > 1000:
        print(f"字体已存在: {FONT_PATH}")
        return True
    for url in FONT_URLS:
        try:
            print(f"尝试下载字体：{url}")
            subprocess.check_call(["wget", "-q", "-O", FONT_PATH, url])
            if os.path.exists(FONT_PATH) and os.path.getsize(FONT_PATH) > 1000:
                print("字体下载成功，刷新字体缓存...")
                try:
                    subprocess.check_call(["fc-cache", "-f", FONT_DIR])
                except Exception:
                    # 有些环境可能没有 fc-cache
                    pass
                return True
        except Exception as e:
            print(f"下载失败: {e}")
    print("未能下载字体，请离线安装中文字体以避免乱码。")
    return False


def parse_markdown_headings(md_text):
    headings = []
    for line in md_text.splitlines():
        m = re.match(r'^(#{1,6})\s+(.*\S)', line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            # 简化：去除代码块标记里的行
            if text.startswith("```"):
                continue
            headings.append((level, text))
    return headings


def build_tree(headings):
    nodes = []
    edges = []
    stack = []  # 存储 (level, node_id)
    for idx, (level, text) in enumerate(headings):
        node_id = f"n{idx}"
        nodes.append((node_id, text))
        # 找父节点：栈中最后一个比当前level小的
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            parent_id = stack[-1][1]
            edges.append((parent_id, node_id))
        stack.append((level, node_id))
    return nodes, edges


def write_dot(nodes, edges, dot_path, fontname="Source Han Sans SC"):
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write('digraph G {\n')
        f.write('  graph [rankdir=LR, fontsize=12];\n')
        f.write(f'  node [shape=box, style=rounded, fontname="{fontname}", fontsize=12];\n')
        f.write(f'  edge [fontname="{fontname}", fontsize=10];\n\n')
        for nid, label in nodes:
            safe_label = label.replace('"', '\\"')
            # 限制过长的标签分行显示
            if len(safe_label) > 80:
                safe_label = safe_label[:77] + "..."
            f.write(f'  {nid} [label="{safe_label}"];\n')
        f.write("\n")
        for a, b in edges:
            f.write(f"  {a} -> {b};\n")
        f.write("}\n")
    print(f"DOT 文件已写入: {dot_path}")


def render_dot(dot_path, png_path):
    try:
        subprocess.check_call(["dot", "-Tpng", "-Gdpi=150", dot_path, "-o", png_path])
        print(f"PNG 已生成: {png_path}")
        return True
    except Exception as e:
        print(f"渲染 PNG 失败（可能未安装 graphviz 二进制）：{e}")
        return False


def main():
    if not os.path.exists(MD_PATH):
        print(f"未找到 Markdown 文件: {MD_PATH}")
        sys.exit(1)
    md_text = open(MD_PATH, "r", encoding="utf-8").read()
    headings = parse_markdown_headings(md_text)
    if not headings:
        print("未从 Markdown 中解析到标题，退出。")
        sys.exit(1)
    nodes, edges = build_tree(headings)
    # 尝试安装字体（非必须）
    try_install_font()
    # 写 dot
    write_dot(nodes, edges, DOT_PATH)
    # 尝试渲染为 png
    rendered = render_dot(DOT_PATH, PNG_PATH)
    if not rendered:
        print("已生成 dot 文件，可在具备 graphviz 的环境中运行：")
        print(f"  dot -Tpng -Gdpi=150 {DOT_PATH} -o {PNG_PATH}")


if __name__ == "__main__":
    main()




