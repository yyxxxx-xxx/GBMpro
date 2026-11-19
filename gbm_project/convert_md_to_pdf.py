#!/usr/bin/env python3
"""
批量将Markdown文件转换为PDF
"""

import os
import sys
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

def convert_markdown_to_pdf(md_file_path, pdf_file_path):
    """将单个Markdown文件转换为PDF"""
    try:
        # 读取Markdown文件
        with open(md_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # 创建PDF对象
        pdf = MarkdownPdf(toc_level=2)

        # 添加内容
        pdf.add_section(Section(markdown_content))

        # 保存PDF
        pdf.save(pdf_file_path)

        print(f"✅ 成功转换: {md_file_path} -> {pdf_file_path}")
        return True

    except Exception as e:
        print(f"❌ 转换失败 {md_file_path}: {e}")
        return False

def main():
    """主函数：批量转换所有指定的Markdown文件"""
    print("🧬 GBM项目报告PDF转换工具")
    print("=" * 50)

    # 要转换的文件列表
    files_to_convert = [
        "gbm_knowledge_method_failure_analysis.md",
        "GBM_BASELINE_COMPARISON_REPORT.md",
        "GBM_KNOWLEDGE_SYSTEM_GUIDE.md",
        "gbm_vs_baseline_theoretical_analysis.md"
    ]

    # 当前目录
    current_dir = Path(__file__).parent

    print(f"📁 工作目录: {current_dir}")
    print(f"📄 待转换文件数量: {len(files_to_convert)}")
    print()

    success_count = 0
    total_count = len(files_to_convert)

    for md_filename in files_to_convert:
        md_file_path = current_dir / md_filename

        # 检查文件是否存在
        if not md_file_path.exists():
            print(f"⚠️  文件不存在: {md_filename}")
            continue

        # 生成PDF文件名
        pdf_filename = md_filename.replace('.md', '.pdf')
        pdf_file_path = current_dir / pdf_filename

        print(f"🔄 正在转换: {md_filename}")

        # 执行转换
        if convert_markdown_to_pdf(md_file_path, pdf_file_path):
            success_count += 1

        print()

    # 转换完成总结
    print("=" * 50)
    print("🎉 转换完成！")
    print(f"✅ 成功: {success_count}/{total_count}")
    print(f"❌ 失败: {total_count - success_count}/{total_count}")

    if success_count > 0:
        print("\n📁 生成的PDF文件:")
        for md_filename in files_to_convert:
            pdf_filename = md_filename.replace('.md', '.pdf')
            pdf_file_path = current_dir / pdf_filename
            if pdf_file_path.exists():
                file_size = pdf_file_path.stat().st_size / 1024  # KB
                print(f"   - {pdf_filename} ({file_size:.1f} KB)")
    if success_count == total_count:
        print("\n🎊 所有文件转换成功！")
    else:
        print(f"\n⚠️  {total_count - success_count} 个文件转换失败")

if __name__ == "__main__":
    main()
