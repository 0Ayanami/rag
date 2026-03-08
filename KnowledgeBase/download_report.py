import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

REPORT_URL = "https://data.eastmoney.com/report/zw_industry.jshtml?infocode=AP202603081820384196"
OUTDIR = "./eastmoney_reports"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


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


def extract_pdf_url(page_url: str, html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    link = soup.select_one("a.pdf-link[href]")
    if link:
        href = (link.get("href") or "").strip()
        if href:
            return urljoin(page_url, href)

    return None


def download_pdf(report_url: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)

    resp = requests.get(report_url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    title = safe_filename(extract_title(soup))
    pdf_url = extract_pdf_url(report_url, html)
    if not pdf_url:
        print("PDF URL: <NOT FOUND>")
        raise RuntimeError("未在页面中找到 PDF 链接。")
    print(f"PDF URL: {pdf_url}")

    pdf_resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
    pdf_resp.raise_for_status()

    output_path = os.path.join(outdir, f"{title}.pdf")
    with open(output_path, "wb") as f:
        f.write(pdf_resp.content)

    return output_path


def main() -> None:
    saved = download_pdf(REPORT_URL, OUTDIR)
    print(f"下载完成: {saved}")


if __name__ == "__main__":
    main()