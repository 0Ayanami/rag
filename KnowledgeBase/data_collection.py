import argparse
from text_extractor import export_pdf_texts
from download_report import run_batch

def build_parser():
    parser = argparse.ArgumentParser(description="从东方财富机构发布页批量下载PDF并导出为Markdown文件")
    parser.add_argument("org_url", help="机构发布列表页 URL，例如: https://data.eastmoney.com/report/orgpublish.jshtml?orgcode=80000031")
    parser.add_argument("--outdir", "-o", default="./data/eastmoney_reports", help="保存目录")
    parser.add_argument("--indexes", default="1-20", help="按序号筛选，例如: 1,2,5-10")
    parser.add_argument("--types", help="按报告类型筛选，逗号分隔")
    parser.add_argument("--targets", help="按研究对象/公司筛选，逗号分隔")
    parser.add_argument("--date-from", help="开始日期（含），格式例子: 2026-01-01")
    parser.add_argument("--date-to", help="结束日期（含")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="并发下载数")
    parser.add_argument("--dry-run", action="store_true", help="只列出将下载的报告，不实际下载")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    # 导出到本地文件的参数
    parser.add_argument("--input", default="data/eastmoney_reports", help="输入路径：可为单个 PDF 文件，或包含 PDF 的目录（默认: data/eastmoney_reports）")
    parser.add_argument("--output", default="data/extracted_md", help="输出目录（默认: data/extracted_md）")
    parser.add_argument("--encoding", default="utf-8", help="Markdown 编码（默认: utf-8）")
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    run_batch(org_url=args.org_url, outdir=args.outdir,
        indexes_arg=args.indexes, types_arg=args.types,
        targets_arg=args.targets, date_from_arg=args.date_from,
        date_to_arg=args.date_to, concurrency=args.concurrency,
        dry_run=args.dry_run, verbose=args.verbose
    )
    count = export_pdf_texts(args.input, args.output, args.encoding)
    print(f"[DONE] 完成提取，成功导出 {count} 个文件到: {args.output}")