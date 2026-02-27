from tqdm import tqdm
import argparse
from pathlib import Path
import re
import pdfplumber

class ProcessPDF:
    def __init__(self, pdf_path: str):
        """解析pdf中的文本信息"""
        self.pdf_path = pdf_path
        self.texts = self.extract_text()

    def extract_text(self):
        """提取文本内容并进行基本清洗"""
        def normalize_text(text: str) -> str:
            if not text:
                return ""
            # 清理中文字符间异常空格
            text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def split_lines_from_words(words, y_tol: float = 2.0):
            lines = []
            current = []
            current_top = None
            for w in sorted(words, key=lambda x: (x["top"], x["x0"])):
                if current_top is None or abs(w["top"] - current_top) <= y_tol:
                    current.append(w)
                    current_top = w["top"] if current_top is None else current_top
                else:
                    lines.append(current)
                    current = [w]
                    current_top = w["top"]
            if current:
                lines.append(current)
            return lines

        def numbered_heading_level(line_text: str) -> int:
            # 匹配类似 "1. ", "1.2 ", "1.2.3 " 的编号标题，返回层级数（最多3级）
            m = re.match(r"^\s*(\d+(?:\.\d+){0,2})(?:[\.、）\)]|\s)+\S+", line_text)
            if not m:
                return 0
            return min(max(len(m.group(1).split(".")), 1), 3)

        full_text = []
        with pdfplumber.open(self.pdf_path) as pdf:
            # 去掉最后一页（通常是免责声明）
            pages = pdf.pages[:-1] if len(pdf.pages) > 1 else pdf.pages

            for page in pages:
                full_text.append(f"<!-- Page {page.page_number} -->")
                words = page.extract_words(
                    x_tolerance=1,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    extra_attrs=["size", "fontname", "top", "x0"],
                ) or []
                lines = split_lines_from_words(words, y_tol=2.0)

                for line_words in lines:
                    line_words = sorted(line_words, key=lambda x: x["x0"])
                    line_text = normalize_text(" ".join(w["text"] for w in line_words))
                    if not line_text:
                        continue

                    level_by_num = numbered_heading_level(line_text)

                    if level_by_num > 0:
                        full_text.append(f"{'#' * level_by_num} {line_text}")
                    else:
                        full_text.append(line_text)
                full_text.append("")
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
        default="../data/extracted_md",
        help="输出目录（默认: ../data/extracted_md）",
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
