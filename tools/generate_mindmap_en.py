#!/usr/bin/env python3
"""
Generate an English mindmap DOT file from RESEARCH_METHODOLOGY.md
 - Parses Markdown headings
 - Maps common Chinese headings to English (fallback to original text)
 - Writes DOT to experiments/mindmaps/llamole_gbm_mindmap_en.dot
 - Attempts to render PNG if graphviz `dot` is available
"""
import os
import re
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD_PATH = os.path.join(ROOT, "gbm_project", "RESEARCH_METHODOLOGY.md")
OUT_DIR = os.path.join(ROOT, "experiments", "mindmaps")
os.makedirs(OUT_DIR, exist_ok=True)
DOT_PATH = os.path.join(OUT_DIR, "llamole_gbm_mindmap_en.dot")
PNG_PATH = os.path.join(OUT_DIR, "llamole_gbm_mindmap_en.png")

# Basic mapping table from Chinese headings (as in the file) to English labels.
TRANSLATION_MAP = {
    "《基于多模态大语言模型的胶质母细胞瘤新药候选分子生成与优化研究》": "Multimodal LLM-based GBM Drug Candidate Generation & Optimization",
    "方法论详细文档": "Methodology Detailed Document",
    "研究概述": "Research Overview",
    "GBM领域知识库": "GBM Knowledge Base",
    "数据来源与构建原则": "Data Sources & Construction Principles",
    "数据来源": "Data Sources",
    "构建原则": "Construction Principles",
    "数据结构与格式": "Data Structure & Format",
    "靶点数据": "Target Data",
    "临床数据": "Clinical Data",
    "分子数据": "Molecular Data",
    "知识库类实现": "Knowledge Base Implementation",
    "GBM专业Prompt生成器": "GBM Prompt Generator",
    "设计理念": "Design Principles",
    "Prompt模板体系": "Prompt Template System",
    "Chain-of-Thought推理模板": "Chain-of-Thought Templates",
    "生成器实现": "Generator Implementation",
    "GBM专项评估器": "GBM Specialized Evaluator",
    "评估体系设计原则": "Evaluation Design Principles",
    "核心评估指标": "Core Evaluation Metrics",
    "血脑屏障穿透性 (BBB Permeability)": "BBB Permeability",
    "GBM细胞活性预测": "GBM Activity Prediction",
    "正常细胞毒性预测": "Normal Cell Toxicity Prediction",
    "选择性指数": "Selectivity Index",
    "合成可行性评分 (SA Score)": "Synthetic Accessibility (SA) Score",
    "临床相似度": "Clinical Similarity",
    "综合得分算法": "Composite Scoring Algorithm",
    "评估器实现": "Evaluator Implementation",
    "GBM分子生成器改进内容": "GBM Molecule Generator Improvements",
    "技术架构改进": "Technical Architecture Improvements",
    "领域知识注入": "Domain Knowledge Injection",
    "Chain-of-Thought推理增强": "Chain-of-Thought Enhancement",
    "多维度约束优化": "Multidimensional Constraint Optimization",
    "智能评估与筛选": "Intelligent Evaluation & Filtering",
    "性能优化策略": "Performance Optimization Strategies",
    "实验验证体系": "Experimental Validation Framework",
    "结论与未来工作": "Conclusions & Future Work",
    "未来研究方向": "Future Research Directions",
}


def parse_markdown_headings(md_text):
    headings = []
    for line in md_text.splitlines():
        m = re.match(r'^(#{1,6})\s+(.*\S)', line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if text.startswith("```"):
                continue
            headings.append((level, text))
    return headings


def translate_label(ch_label):
    # Direct map
    if ch_label in TRANSLATION_MAP:
        return TRANSLATION_MAP[ch_label]
    # Remove trailing colons or parentheses with content
    cleaned = re.sub(r'\s*[:：]\s*$', '', ch_label)
    if cleaned in TRANSLATION_MAP:
        return TRANSLATION_MAP[cleaned]
    # Try short heuristics
    heur = cleaned
    heur = heur.replace("（", "(").replace("）", ")")
    if heur in TRANSLATION_MAP:
        return TRANSLATION_MAP[heur]
    # Fallback: return original (Chinese)
    return ch_label


def build_tree(headings):
    nodes = []
    edges = []
    stack = []
    for idx, (level, text) in enumerate(headings):
        node_id = f"n{idx}"
        en_label = translate_label(text)
        nodes.append((node_id, en_label))
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            parent_id = stack[-1][1]
            edges.append((parent_id, node_id))
        stack.append((level, node_id))
    return nodes, edges


def write_dot(nodes, edges, dot_path):
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write('digraph G {\n')
        f.write('  graph [rankdir=LR, fontsize=12];\n')
        f.write('  node [shape=box, style=rounded, fontname="Helvetica", fontsize=12];\n')
        f.write('  edge [fontname="Helvetica", fontsize=10];\n\n')
        for nid, label in nodes:
            safe_label = label.replace('"', '\\"')
            if len(safe_label) > 80:
                safe_label = safe_label[:77] + "..."
            f.write(f'  {nid} [label="{safe_label}"];\n')
        f.write("\n")
        for a, b in edges:
            f.write(f"  {a} -> {b};\n")
        f.write("}\n")
    print(f"English DOT written to: {dot_path}")


def render_dot(dot_path, png_path):
    try:
        subprocess.check_call(["dot", "-Tpng", "-Gdpi=150", dot_path, "-o", png_path])
        print(f"PNG generated: {png_path}")
        return True
    except FileNotFoundError:
        print("dot not found (graphviz not installed). DOT created for offline rendering.")
        return False
    except Exception as e:
        print(f"Failed to render PNG: {e}")
        return False


def main():
    if not os.path.exists(MD_PATH):
        print("Markdown not found:", MD_PATH)
        sys.exit(1)
    md_text = open(MD_PATH, "r", encoding="utf-8").read()
    headings = parse_markdown_headings(md_text)
    if not headings:
        print("No headings parsed.")
        sys.exit(1)
    nodes, edges = build_tree(headings)
    write_dot(nodes, edges, DOT_PATH)
    rendered = render_dot(DOT_PATH, PNG_PATH)
    if not rendered:
        print("To render PNG locally, run:")
        print(f"  dot -Tpng -Gdpi=150 {DOT_PATH} -o {PNG_PATH}")


if __name__ == "__main__":
    main()



