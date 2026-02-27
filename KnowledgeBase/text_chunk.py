import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


@dataclass
class ChunkConfig:
    method: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 50
    break_threshold: float = 0.3
    ollama_model: str = "qwen3-embedding"
    ollama_base_url: str = "http://localhost:11434"


def collect_text_files(input_path: str, suffixes: Sequence[str] = (".md", ".txt")) -> List[Path]:
    path = Path(input_path)
    normalized = tuple(s.lower() for s in suffixes)
    if path.is_file() and path.suffix.lower() in normalized:
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in normalized])
    return []


def load_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(input_path: str, suffixes: Sequence[str] = (".md", ".txt")) -> List[Document]:
    files = collect_text_files(input_path, suffixes=suffixes)
    return [
        Document(
            page_content=load_text(p),
            metadata={"source": str(p), "filename": p.name, "suffix": p.suffix.lower()},
        )
        for p in files
    ]


def _char_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
    )


def chunk_fixed(documents: Iterable[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    return _char_splitter(chunk_size, chunk_overlap).split_documents(list(documents))


def chunk_markdown_header_recursive(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
        strip_headers=False,
    )
    splitter = _char_splitter(chunk_size, chunk_overlap)
    chunks: List[Document] = []

    for doc in documents:
        if str(doc.metadata.get("suffix", "")).lower() != ".md":
            chunks.extend(splitter.split_documents([doc]))
            continue

        section_docs = header_splitter.split_text(doc.page_content)
        if not section_docs:
            chunks.extend(splitter.split_documents([doc]))
            continue

        for sec in section_docs:
            meta = dict(doc.metadata)
            meta.update(sec.metadata or {})
            chunks.extend(splitter.create_documents([sec.page_content], metadatas=[meta]))

    return chunks


def chunk_semantic(
    documents: Iterable[Document], break_threshold: float, ollama_model: str, ollama_base_url: str
) -> List[Document]:
    embeddings = OllamaEmbeddings(model=ollama_model, base_url=ollama_base_url)
    splitter = SemanticChunker(
        embeddings=embeddings,
        sentence_split_regex=r"(?<=[。！？!?；;.\n])\s*",
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=max(1, min(99, int(break_threshold * 100))),
    )
    return splitter.split_documents(list(documents))


def chunk_documents(documents: Iterable[Document], config: ChunkConfig) -> List[Document]:
    method = config.method.lower().strip()
    if method == "fixed":
        return chunk_fixed(documents, config.chunk_size, config.chunk_overlap)
    if method in {"header", "markdown_header_recursive", "markdown"}:
        return chunk_markdown_header_recursive(documents, config.chunk_size, config.chunk_overlap)
    if method == "semantic":
        return chunk_semantic(
            documents,
            break_threshold=config.break_threshold,
            ollama_model=config.ollama_model,
            ollama_base_url=config.ollama_base_url,
        )
    raise ValueError(f"Unsupported chunk method: {config.method}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取 markdown/txt 文档并分块")
    parser.add_argument("--input", required=True, help="输入路径：单文件或目录")
    parser.add_argument(
        "--method",
        default="fixed",
        choices=["fixed", "semantic", "markdown_header_recursive"],
        help="分块策略",
    )
    parser.add_argument("--chunk_size", type=int, default=500, help="固定/标题递归分块大小")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="固定/标题递归分块重叠长度")
    parser.add_argument("--break_threshold", type=float, default=0.3, help="语义分块断点阈值")
    parser.add_argument("--ollama_model", type=str, default="qwen3-embedding", help="本地 Ollama Embedding 模型名")
    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434", help="Ollama 服务地址")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    docs = load_documents(args.input, suffixes=(".md", ".txt"))
    cfg = ChunkConfig(
        method=args.method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        break_threshold=args.break_threshold,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
    )
    chunks = chunk_documents(docs, cfg)
    print(f"[DONE] files={len(docs)} chunks={len(chunks)} method={cfg.method}")
    for i, c in enumerate(chunks[:5], start=1):
        src = c.metadata.get("filename", c.metadata.get("source", "unknown"))
        preview = c.page_content[:100].replace("\n", " ")
        print(f"{i:02d}. {src} -> {preview}...")
