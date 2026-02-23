import os
import re
import time
import requests
import pdfplumber
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

ORG_PAGE = "https://data.eastmoney.com/report/orgpublish.jshtml?orgcode=80000031"
OUTDIR = "./eastmoney_reports"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}
# 可按需调整：最大等待时间、每次请求间隔（秒）
SELENIUM_WAIT = 12
REQUEST_SLEEP = 1.2

os.makedirs(OUTDIR, exist_ok=True)

def safe_filename(s):
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r'\s+', " ", s).strip()
    if len(s) > 200:
        s = s[:200]
    return s

def get_report_links_with_selenium(org_url):
    """用 Selenium 渲染机构页并抓取指向研报详情页的链接（infocode / zw_stock 等模式）"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # 可按需添加代理等
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(org_url)
    try:
        # 等待页面加载出若干 a 元素（因为数据通过 JS 加载）
        WebDriverWait(driver, SELENIUM_WAIT).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )
    except Exception as e:
        print("等待页面加载超时（但会尽力抓取当前已加载的内容）:", e)

    anchors = driver.find_elements(By.TAG_NAME, "a")
    hrefs = set()
    for a in anchors:
        try:
            href = a.get_attribute("href")
            if not href:
                continue
            # 过滤：我们关心包含 infocode 或 /report/zw_stock.jshtml 的详情页
            if "infocode=" in href or "/report/zw_stock.jshtml" in href or "/report/zw.jshtml" in href:
                hrefs.add(href.split("#")[0])
        except Exception:
            continue
    driver.quit()
    return sorted(hrefs)

def extract_text_from_html(url, text_html):
    """解析HTML页面并抽取正文"""
    soup = BeautifulSoup(text_html, "lxml")

    # 尝试获取标题
    title = None
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    if not title:
        # 备用取法
        title_tag = soup.find(attrs={"class": re.compile(r"title|report-title|zw-title", re.I)})
        if title_tag:
            title = title_tag.get_text(strip=True)
    if not title and soup.title:
        title = soup.title.get_text(strip=True)
    if not title:
        # 最后兜底
        title = urlparse(url).path.split("/")[-1]

    full = soup.get_text("\n", strip=True)
    start_idx = full.find(title)
    if start_idx == -1:
        start_idx = 0
    # 常见结束标识
    end_markers = ["郑重声明", "免责声明", "今日最新研究报告", "数据来源：", "Information来源", "— END —"]
    end_idx = None
    for m in end_markers:
        idx = full.find(m, start_idx + 1)
        if idx != -1:
            if end_idx is None or idx < end_idx:
                end_idx = idx
    content = full[start_idx:end_idx].strip() if end_idx else full[start_idx:].strip()
    return title, content

def download_pdf_and_extract_text(pdf_url):
    """若页面提供 PDF 链接，下载并尝试用 pdfplumber 提取文本（作为备用）"""
    try:
        r = requests.get(pdf_url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        tmpfn = os.path.join(OUTDIR, "tmp_report.pdf")
        with open(tmpfn, "wb") as f:
            f.write(r.content)
        text_parts = []
        with pdfplumber.open(tmpfn) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        os.remove(tmpfn)
        return "\n\n".join(text_parts).strip()
    except Exception as e:
        print("PDF 处理出错:", e)
        return None

def fetch_and_save_report(url):
    """获取单篇研报：优先抓 HTML 正文；若页面只给 PDF，尝试下载并提取；最后保存 txt"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
    except Exception as e:
        print("请求失败:", url, e)
        return False
    if r.status_code != 200:
        print("非 200:", r.status_code, url)
        return False

    # 先在 HTML 中查找 PDF 链接（如果页面本身是 HTML 并展示正文，则优先用 HTML）
    soup = BeautifulSoup(r.text, "lxml")
    # 找正文启发式
    title, content = extract_text_from_html(url, r.text)
    if content and len(content) > 50:
        filename = safe_filename(title) + ".txt"
        path = os.path.join(OUTDIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n")
            f.write(content)
        return True

    # HTML 正文不足时，检查是否有 PDF 链接（常见域：pdf.dfcfw.com / pdf.eastmoney.com 等）
    pdf_link = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf") or "pdf." in href or "dfcfw.com" in href:
            pdf_link = urljoin(url, href)
            break
    if pdf_link:
        pdf_text = download_pdf_and_extract_text(pdf_link)
        if pdf_text:
            filename = safe_filename(title) + ".txt"
            path = os.path.join(OUTDIR, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{title}\n\n")
                f.write(pdf_text)
            return True

    # 最后兜底：把 HTML 的主要文本（全部）保存
    whole_text = soup.get_text("\n", strip=True)
    filename = safe_filename(title) + ".txt"
    path = os.path.join(OUTDIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        f.write(whole_text)
    return True

def main():
    print("开始用 Selenium 获取详情页链接（可能需要下载 ChromeDriver）...")
    links = get_report_links_with_selenium(ORG_PAGE)
    print(f"共抓取到 {len(links)} 个可能的详情页链接（去重/过滤后）")
    # 过滤同站内有效的详情页（只保留 data.eastmoney.com 下有 infocode 或 zw_stock）
    filtered = []
    for u in links:
        if "data.eastmoney.com" in u and ("infocode" in u or "zw_stock" in u or "zw.jshtml" in u):
            filtered.append(u)
    filtered = sorted(set(filtered))
    print(f"筛选后 {len(filtered)} 条详情页将被下载（如果太多可在代码中限制数量）")

    # 如果太多，下面可以设置一个 limit，例如 filtered = filtered[:200]
    for u in tqdm(filtered, desc="下载研报"):
        try:
            success = fetch_and_save_report(u)
            if not success:
                print("保存失败：", u)
        except Exception as e:
            print("处理异常：", u, e)
        time.sleep(REQUEST_SLEEP)

if __name__ == "__main__":
    main()