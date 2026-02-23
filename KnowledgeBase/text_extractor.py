from tqdm import tqdm
import os
import argparse
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

# 文本切分器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

class ProcessPDF:
    def __init__(self, pdf_path: str):
        """解析pdf中的文本信息"""
        self.pdf_path = pdf_path
        self.texts = self.extract_text()

    def extract_text(self):
        """提取全部文本内容（含基础排版信息）"""
        full_text = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                # 提取当前页文本（保留Y坐标顺序）
                text = page.extract_text(x_tolerance=1, y_tolerance=3)
                full_text.append(f"=== Page {page.page_number} ===")
                full_text.append(text)
        return "\n".join(full_text)


def collect_pdf_files(input_path: str):
    """返回包含所有子文件的完整路径列表"""
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.pdf"))
    return []


def export_pdf_texts(input_path: str, output_dir: str, encoding: str = "utf-8"):
    pdf_files = collect_pdf_files(input_path)
    if not pdf_files:
        print(f"[WARN] 未找到 PDF 文件: {input_path}")
        return 0

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(input_path)

    success_count = 0
    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        try:
            text = ProcessPDF(str(pdf_file)).texts or ""
            if input_root.is_dir():
                rel_pdf = pdf_file.relative_to(input_root)
                output_file = out_root / rel_pdf.with_suffix(".md")
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = out_root / f"{pdf_file.stem}.md"
            output_file.write_text(text, encoding=encoding)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] 处理失败: {pdf_file} -> {e}")

    return success_count


def build_parser():
    parser = argparse.ArgumentParser(description="提取 PDF 文本并导出为 Markdown 文件")
    parser.add_argument(
        "--input",
        required=True,
        help="输入路径：可为单个 PDF 文件，或包含 PDF 的目录",
    )
    parser.add_argument(
        "--output",
        default="./extracted_md",
        help="输出目录（默认: ./extracted_md）",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Markdown 编码（默认: utf-8）",
    )
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    count = export_pdf_texts(args.input, args.output, args.encoding)
    print(f"[DONE] 完成提取，成功导出 {count} 个文件到: {args.output}")
