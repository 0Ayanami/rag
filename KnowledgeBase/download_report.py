"""
批量抓取东方财富某机构发布页的报告并下载 PDF。
Features:
- 解析页面中 body -> main -> main-content -> framecontent -> org_publishtable 的 table（通过 id/name/class 尝试定位）
- 按序号、报告类型、研究对象、日期区间筛选
- 并发下载（ThreadPoolExecutor）
- 多种方式查找 PDF 链接（a[href$=.pdf], iframe/embed/object 指向 pdf, 自定义 a.pdf-link 等）
- 支持 dry-run 模式打印将下载的项
"""

import os
import re
import json
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Dict, Any

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
DEFAULT_TIMEOUT = 20


def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200] if len(name) > 200 else name


def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
        title = re.sub(r"[\-_]\s*东方财富网.*$", "", title).strip()
        return title or "eastmoney_report"

    return "eastmoney_report"


def extract_pdf_url(page_url: str, html: str) -> Optional[str]:
    """
    从报告页面 HTML 中尝试多种策略找到 PDF 链接，返回绝对 URL 或 None
    """
    soup = BeautifulSoup(html, "lxml")

    # 常见直接 <a href="...pdf">
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if re.search(r"\.pdf($|\?)", href, flags=re.I):
            return urljoin(page_url, href)

    # 特定类名或选择器（如 a.pdf-link）
    link = soup.select_one("a.pdf-link[href]")
    if link:
        return urljoin(page_url, link["href"].strip())
    return None


def download_pdf_from_report(report_page_url: str, outdir: str, session: requests.Session, verbose: bool = False) -> Optional[str]:
    """
    访问报告详情页，寻找 PDF 并下载。返回保存路径或 None。
    """
    try:
        resp = session.get(report_page_url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        if verbose:
            print(f"[ERROR] 请求报告页失败: {report_page_url} -> {e}")
        return None

    pdf_url = extract_pdf_url(report_page_url, resp.text)
    if not pdf_url:
        if verbose:
            print(f"[WARN] 未在报告页找到 PDF 链接: {report_page_url}")
        return None

    try:
        pdf_resp = session.get(pdf_url, headers=HEADERS, timeout=DEFAULT_TIMEOUT * 2)
        pdf_resp.raise_for_status()
    except Exception as e:
        if verbose:
            print(f"[ERROR] 下载 PDF 失败: {pdf_url} -> {e}")
        return None

    try:
        soup = BeautifulSoup(resp.text, "lxml")
        title = safe_filename(extract_title(soup))
    except Exception:
        title = safe_filename(os.path.basename(urlparse(pdf_url).path) or "report")

    filename = f"{title}.pdf"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)

    i = 1
    base, ext = os.path.splitext(path)
    while os.path.exists(path):
        path = f"{base}_{i}{ext}"
        i += 1

    with open(path, "wb") as f:
        f.write(pdf_resp.content)

    if verbose:
        print(f"[OK] 下载完成: {path}")
    return path


def find_org_table(soup: BeautifulSoup) -> Optional[Any]:
    """
    尝试定位 orgpublish_table
    """
    # 精确路径优先
    selectors = [
        "html body div.main div.main-content div.framecontent div#orgpublish_table table.table-model",
        "body div.main div.main-content div.framecontent div#orgpublish_table table.table-model",
        "div.main div.main-content div.framecontent div#orgpublish_table table.table-model",
        "#orgpublish_table table.table-model",
        "#orgpublish_table table",
        "#org_publishtable table.table-model",  # 页面里偶见旧命名
        "#org_publishtable table",
        "table.table-model",
    ]
    for selector in selectors:
        table = soup.select_one(selector)
        if table:
            return table
    return soup.find("table")


def parse_rows_from_initdata(html: str, base_url: str) -> List[Dict[str, str]]:
    """
    页面无静态 table 时，从 `var initdata = {...};` 解析首屏数据。
    """
    m = re.search(r"var\s+initdata\s*=\s*(\{.*?\});", html, flags=re.S)
    if not m:
        return []

    try:
        payload = json.loads(m.group(1))
    except Exception:
        return []

    data = payload.get("data") or []
    rows: List[Dict[str, str]] = []
    type_map = {2: "策略报告", 3: "行业研究", 4: "券商晨会"}

    for i, item in enumerate(data, 1):
        info_code = str(item.get("infoCode") or "").strip()
        report_url = urljoin(base_url, f"/report/info/{info_code}.html") if info_code else ""

        author = item.get("researcher") or ""
        if not author:
            a = item.get("author")
            if isinstance(a, list):
                author = ",".join([str(x) for x in a if x is not None])
            elif a is not None:
                author = str(a)

        publish_date = str(item.get("publishDate") or "")
        if publish_date:
            publish_date = publish_date.split(" ")[0]

        report_type_value = item.get("reportType")
        report_type = str(item.get("columnType") or "").strip()
        if not report_type:
            report_type = type_map.get(report_type_value, str(report_type_value or ""))

        target = item.get("stockName") or item.get("industryName") or ""

        rows.append(
            {
                "index": str(i),
                "title": str(item.get("title") or ""),
                "type": str(report_type),
                "target": str(target),
                "author": str(author),
                "org": str(item.get("orgName") or ""),
                "date": publish_date,
                "report_url": report_url,
            }
        )

    return rows

def parse_table_rows(table, base_url: str) -> List[Dict[str, str]]:
    """
    解析表格，返回每行的字典：
    {
      "index": ..., 
      "title": ...,         # 报告名称（从该列提取文本）
      "type": ...,          # 报告类型
      "target": ...,        # 研究对象
      "author": ...,        # 作者
      "org": ...,           # 机构
      "date": ...,          # 日期
      "report_url": ...     # 报告详情页（绝对 URL，尽可能提取）
    }
    base_url 用于把相对 href 变成绝对 URL。
    """
    # 先尝试读取表头（thead 优先）
    headers = []
    header_row = None
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
    if not header_row:
        first_tr = table.find("tr")
        header_row = first_tr

    if header_row:
        for th in header_row.find_all("th"):
            headers.append(th.get_text(strip=True).lower())

    print(f"[DEBUG] 表头列数: {len(headers)}，内容: {headers}")

    # 定义按关键字匹配列索引的 helper
    def col_idx_by_keywords(keywords):
        for kw in keywords:
            for i, h in enumerate(headers):
                if kw in h:
                    return i
        return None

    # 根据你提供的顺序优先映射
    idx_col = col_idx_by_keywords(["序号", "id", "no"])
    title_col = col_idx_by_keywords(["报告名称", "标题", "name"])
    type_col = col_idx_by_keywords(["报告类型", "类型", "report type"]) 
    target_col = col_idx_by_keywords(["研究对象", "对象", "company"])
    author_col = col_idx_by_keywords(["作者", "撰写", "author"])
    org_col = col_idx_by_keywords(["机构", "发布机构", "机构名称"])
    date_col = col_idx_by_keywords(["日期", "发布时间", "发布日期", "time", "date"])

    rows = []
    tbody = table.find("tbody") or table

    # 如果 header_row 是 table 中的第一行并且没有 thead，需要在解析数据时跳过它
    skip_first = (table.find("thead") is None) and header_row is not None

    for i, tr in enumerate(tbody.find_all("tr")):
        if skip_first and i == 0:
            # 跳过 header 行
            continue

        tds = tr.find_all(["td", "th"])
        if not tds:
            continue

        def cell_text(col):
            if col is None:
                return ""
            if col < len(tds):
                return tds[col].get_text(strip=True)
            return ""

        # 报告链接：优先从 title_col 的 <a> 中提取
        report_url = ""
        title_text = ""
        if title_col is not None and title_col < len(tds):
            title_cell = tds[title_col]
            a = title_cell.find("a", href=True)
            if a:
                href = a["href"].strip()
                report_url = urljoin(base_url, href)
                title_text = a.get_text(strip=True)
            else:
                # 没有 a 的话取整列文本
                title_text = title_cell.get_text(strip=True)

        # 如果没有在 title_col 找到链接，退回到整行搜索第一个 a[href]
        if not report_url:
            a_row = tr.find("a", href=True)
            if a_row:
                report_url = urljoin(base_url, a_row["href"].strip())
                if not title_text:
                    title_text = a_row.get_text(strip=True)

        # fallback: 若 title_text 仍然为空，使用列文本或空串
        if not title_text and title_col is not None:
            title_text = cell_text(title_col)

        row = {
            "index": cell_text(idx_col),
            "title": title_text,
            "type": cell_text(type_col),
            "target": cell_text(target_col),
            "author": cell_text(author_col),
            "org": cell_text(org_col),
            "date": cell_text(date_col),
            "report_url": report_url
        }
        rows.append(row)

    return rows

def parse_date(datestr: str) -> Optional[datetime]:
    if not datestr:
        return None
    datestr = datestr.strip()
    # 常见格式尝试
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y年%m月%d日",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(datestr, fmt)
        except Exception:
            continue
    # 尝试只抽取 YYYY-MM-DD
    m = re.search(r"(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})", datestr)
    if m:
        s = m.group(1).replace(".", "-").replace("/", "-")
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            pass
    # 最后尝试仅取年份
    m2 = re.search(r"(\d{4})", datestr)
    if m2:
        try:
            return datetime(int(m2.group(1)), 1, 1)
        except Exception:
            pass
    return None


def match_filters(row: Dict[str, str], filters: Dict[str, Any]) -> bool:
    if filters.get("indexes"):
        try:
            # row["index"] may include non-digits; extract digits
            digits = re.search(r"(\d+)", row.get("index", ""))
            if not digits:
                return False
            idx = int(digits.group(1))
            if idx not in filters["indexes"]:
                return False
        except Exception:
            return False

    if filters.get("types"):
        v = row.get("type", "")
        ok = False
        for t in filters["types"]:
            if t in v:
                ok = True
                break
        if not ok:
            return False

    if filters.get("targets"):
        v = row.get("target", "")
        ok = False
        for t in filters["targets"]:
            if t in v:
                ok = True
                break
        if not ok:
            return False

    if filters.get("date_from") or filters.get("date_to"):
        d = parse_date(row.get("date", "")) or None
        if d is None:
            return False
        if filters.get("date_from") and d < filters["date_from"]:
            return False
        if filters.get("date_to") and d > filters["date_to"]:
            return False

    return True


def parse_indexes_arg(arg: Optional[str]) -> Optional[set]:
    if not arg:
        return None
    s = set()
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                a_i = int(a)
                b_i = int(b)
                for i in range(min(a_i, b_i), max(a_i, b_i) + 1):
                    s.add(i)
            except Exception:
                continue
        else:
            try:
                s.add(int(p))
            except Exception:
                continue
    return s if s else None


def run_batch(org_url: str, outdir: str, indexes_arg: Optional[str], types_arg: Optional[str],
              targets_arg: Optional[str], date_from_arg: Optional[str], date_to_arg: Optional[str],
              concurrency: int = 4, dry_run: bool = False, verbose: bool = False):
    session = requests.Session()
    try:
        resp = session.get(org_url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"请求机构页面失败: {org_url} -> {e}")

    base_url = "https://data.eastmoney.com"
    soup = BeautifulSoup(resp.text, "lxml")
    table = find_org_table(soup)
    rows: List[Dict[str, str]] = []
    if table:
        rows = parse_table_rows(table, base_url)
        if verbose:
            print(f"[INFO] 通过 HTML table 解析到 {len(rows)} 行")

    # 页面常见 JS 渲染：无 table 时改从 initdata 提取
    if not rows:
        rows = parse_rows_from_initdata(resp.text, base_url)
        if verbose:
            print(f"[INFO] 通过 initdata 解析到 {len(rows)} 行")

    if not rows:
        raise RuntimeError("未找到可解析的数据（table 与 initdata 均为空），请检查页面结构或网络返回内容。")

    if verbose:
        print(f"[INFO] 总解析行数: {len(rows)}")

    filters = {}
    idx_set = parse_indexes_arg(indexes_arg)
    if idx_set:
        filters["indexes"] = idx_set
    if types_arg:
        filters["types"] = [t.strip() for t in types_arg.split(",") if t.strip()]
    if targets_arg:
        filters["targets"] = [t.strip() for t in targets_arg.split(",") if t.strip()]
    if date_from_arg:
        filters["date_from"] = parse_date(date_from_arg)
    if date_to_arg:
        filters["date_to"] = parse_date(date_to_arg)

    # prepare tasks
    tasks = []
    base = org_url
    for r in rows:
        # some report_url may be javascript or empty; skip those without href
        href = r.get("report_url", "")
        if not href:
            continue
        # normalize to absolute
        abs_href = urljoin(base, href)
        r["report_url"] = abs_href
        if match_filters(r, filters):
            tasks.append(r)

    if not tasks:
        print("[INFO] 没有找到匹配条件的报告条目。")
        return

    print(f"[INFO] 将下载 {len(tasks)} 个报告（concurrency={concurrency}，dry_run={dry_run}）")
    if dry_run:
        for t in tasks:
            print(f" - index={t.get('index')} date={t.get('date')} type={t.get('type')} target={t.get('target')} url={t.get('report_url')}")
        return

    # download with thread pool
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        future_to_row = {}
        for r in tasks:
            future = ex.submit(download_pdf_from_report, r["report_url"], outdir, session, verbose)
            future_to_row[future] = r

        for fut in as_completed(future_to_row):
            r = future_to_row[fut]
            try:
                path = fut.result()
                results.append((r, path))
            except Exception as e:
                print(f"[ERROR] 下载任务异常: {r.get('report_url')} -> {e}")
                results.append((r, None))

    # summary
    ok = sum(1 for _, p in results if p)
    fail = len(results) - ok
    print(f"[DONE] 成功 {ok}，失败 {fail}，保存目录: {os.path.abspath(outdir)}")
    if verbose and fail > 0:
        for r, p in results:
            if not p:
                print(f"  [FAIL] index={r.get('index')} url={r.get('report_url')}")


def main_cli():
    parser = argparse.ArgumentParser(description="从东方财富机构发布页批量下载 PDF（支持筛选）")
    parser.add_argument("org_url", help="机构发布列表页 URL，例如: https://data.eastmoney.com/report/orgpublish.jshtml?orgcode=80000031")
    parser.add_argument("--outdir", "-o", default="./eastmoney_reports", help="保存目录")
    parser.add_argument("--indexes", default="1-20", help="按序号筛选，例如: 1,2,5-10")
    parser.add_argument("--types", help="按报告类型筛选，逗号分隔，支持部分匹配（中文）")
    parser.add_argument("--targets", help="按研究对象/公司筛选，逗号分隔，支持部分匹配（中文）")
    parser.add_argument("--date-from", help="开始日期（含），格式例子: 2026-01-01 或 2026/01/01 或 2026年01月01日")
    parser.add_argument("--date-to", help="结束日期（含）")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="并发下载数")
    parser.add_argument("--dry-run", action="store_true", help="只列出将下载的报告，不实际下载")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    run_batch(
        org_url=args.org_url,
        outdir=args.outdir,
        indexes_arg=args.indexes,
        types_arg=args.types,
        targets_arg=args.targets,
        date_from_arg=args.date_from,
        date_to_arg=args.date_to,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main_cli()
