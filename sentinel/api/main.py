"""
FastAPI Server for Sentinel.
Exposes the core quantitative engines via high-throughput REST endpoints.
"""

import io
import json
import logging
import os
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from sentinel.db.database import Base, engine, get_db
from sentinel.db.models import Borrower, Loan, Transaction

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema if it doesn't exist
    Base.metadata.create_all(bind=engine)

    # Initialize meta.json if it doesn't exist
    meta_path = "sentinel.db.meta.json"
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump({"type": "synthetic", "desc": "Default Seed (Medium | Normal Baseline | 0.2% Fraud)"}, f)

    yield

app = FastAPI(title="Sentinel Risk Intelligence API", lifespan=lifespan)

# --- 1. CORE HTML TEMPLATE ---

def get_base_html(content: str, active_tab: str, breadcrumb: str):
    tabs = [
        ("home", "layout-dashboard", "Home", ""),
        ("fraud", "shield-alert", "Transaction Surveillance", "Risk Engines"),
        ("credit", "briefcase", "Loan Default Engine", "Risk Engines"),
        ("market", "line-chart", "Portfolio VaR", "Risk Engines"),
        ("volatility", "activity", "Volatility Surface", "Risk Engines"),
        ("stress", "flame", "Scenario Analysis", "Risk Engines"),
        ("customer_360", "user-check", "Obligor 360", "Risk Engines"),
        ("database", "table-2", "Risk Datamart", "Data Infrastructure"),
        ("generator", "cpu", "Data Forge", "Data Infrastructure"),
        ("importer", "upload-cloud", "Data Ingestion", "Data Infrastructure"),
    ]

    nav_html = ""
    current_parent = "UNSET_PARENT"
    for id, icon, name, parent in tabs:
        if parent != current_parent:
            if parent:
                mt_class = "mt-6" if current_parent != "UNSET_PARENT" else "mt-2"
                nav_html += f'<div class="px-3 {mt_class} mb-2 text-[10px] font-bold tracking-wider text-gray-400 dark:text-gray-500 uppercase">{parent}</div>'
            current_parent = parent

        active_class = "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white font-semibold" if active_tab == id else "text-gray-600 dark:text-gray-400 hover:bg-gray-200/50 dark:hover:bg-gray-700/50 hover:text-gray-900 dark:text-white"
        nav_html += f"""
            <div class="px-3">
                <button hx-get="/ui/view/{id}" hx-target="#workspace" hx-push-url="true" onclick="updateNav('{id}', '{parent}', '{name}')" id="nav-{id}" class="nav-btn w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors text-left {active_class}">
                    <i data-lucide="{icon}" class="w-4 h-4"></i> {name}
                </button>
            </div>
        """

    header_parent = breadcrumb.split(" > ")[0]
    header_child = breadcrumb.split(" > ")[1] if " > " in breadcrumb else ""

    # We will compute the breadcrumb HTML in python to avoid f-string JS conflicts
    if header_child:
        breadcrumb_html = f'<span id="header-parent" class="">{header_parent}</span><i id="header-chevron" data-lucide="chevron-right" class="w-4 h-4 text-gray-300 dark:text-gray-600"></i><span id="header-child" class="text-gray-900 dark:text-white font-semibold">{header_child}</span>'
    else:
        # Show top level parent if no child (like Home)
        breadcrumb_html = f'<span id="header-parent" class="text-gray-900 dark:text-white font-semibold">{header_parent}</span><i id="header-chevron" data-lucide="chevron-right" class="w-4 h-4 text-gray-300 dark:text-gray-600" style="display:none;"></i><span id="header-child" class="text-gray-900 dark:text-white font-semibold" style="display:none;"></span>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel Risk Intelligence</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.tailwindcss.com?plugins=forms"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{}}
            }}
        }}
    </script>
    <script>
        if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
            document.documentElement.classList.add('dark');
        }} else {{
            document.documentElement.classList.remove('dark');
        }}
        function toggleTheme() {{
            document.documentElement.classList.toggle('dark');
            localStorage.theme = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
        }}
    </script>
    <style>
        @keyframes marquee {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        .animate-marquee {{ animation: marquee 30s linear infinite; }}
        .htmx-indicator {{ display:none; }}
        .htmx-request .htmx-indicator {{ display:flex; }}
    </style>
</head>
<body class="bg-gray-50 dark:bg-gray-800 h-screen flex overflow-hidden font-sans text-gray-900 dark:text-white">
    <div class="w-64 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex flex-col shrink-0">
        <div class="h-16 flex items-center px-6 border-b border-gray-200 dark:border-gray-700 shrink-0">
            <div class="flex items-center gap-2">
                <div class="w-8 h-8 rounded bg-gray-900 flex items-center justify-center text-white font-bold"><i data-lucide="shield" class="w-5 h-5"></i></div>
                <span class="font-bold text-lg tracking-tight">Sentinel</span>
            </div>
        </div>
        <div class="flex-1 overflow-y-auto py-4">
            {nav_html}
        </div>
    </div>

    <div class="flex-1 flex flex-col min-w-0">
        <header class="h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-8 shrink-0 min-w-0">
            <div class="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400 font-medium">
                {breadcrumb_html}
            </div>
            
            <div class="flex-1 overflow-hidden ml-8 flex items-center bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg h-9 min-w-0" hx-get="/ui/widget/ticker" hx-trigger="load, every 60s">
                <div class="text-xs text-gray-400 dark:text-gray-500 px-4">Loading market rates...</div>
            </div>
            
            <button onclick="toggleTheme()" class="ml-4 p-2 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                <i data-lucide="moon" class="w-5 h-5 hidden dark:block"></i>
                <i data-lucide="sun" class="w-5 h-5 block dark:hidden"></i>
            </button>
        </header>
        
        <div class="flex-1 overflow-y-auto p-8 bg-gray-50/50 dark:bg-gray-800/50">
            <div id="workspace" class="max-w-7xl mx-auto transition-all duration-300">
                {content}
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();
        document.body.addEventListener('htmx:afterSwap', function(evt) {{
            lucide.createIcons();
        }});
        
        function updateNav(id, parent, name) {{
            document.querySelectorAll('.nav-btn').forEach(btn => {{
                btn.classList.remove('bg-gray-100', 'dark:bg-gray-800', 'text-gray-900', 'dark:text-white', 'font-semibold');
                btn.classList.add('text-gray-600', 'dark:text-gray-400');
            }});
            const active = document.getElementById('nav-' + id);
            active.classList.add('bg-gray-100', 'dark:bg-gray-800', 'text-gray-900', 'dark:text-white', 'font-semibold');
            active.classList.remove('text-gray-600', 'dark:text-gray-400');
            
            if (parent === "") {{
                document.getElementById('header-parent').textContent = name;
                document.getElementById('header-parent').classList.add('text-gray-900', 'dark:text-white', 'font-semibold');
                if (document.getElementById('header-chevron')) document.getElementById('header-chevron').style.display = 'none';
                if (document.getElementById('header-child')) document.getElementById('header-child').style.display = 'none';
            }} else {{
                document.getElementById('header-parent').textContent = parent;
                document.getElementById('header-parent').classList.remove('text-gray-900', 'dark:text-white', 'font-semibold');
                if (document.getElementById('header-chevron')) document.getElementById('header-chevron').style.display = 'inline-block';
                if (document.getElementById('header-child')) {{
                    document.getElementById('header-child').style.display = 'inline-block';
                    document.getElementById('header-child').textContent = name;
                }}
            }}
        }}
    </script>
</body>
</html>
"""

# --- 2. ROOT & TICKER ---

@app.get("/")
def root(request: Request, db: Session = Depends(get_db)):
    return view_home(request, db)

@app.get("/ui/widget/ticker", response_class=HTMLResponse)
def get_ticker():
    rates = [
        ("SOFR", "5.31%", "+0.01"),
        ("US 10Y", "4.25%", "-0.04"),
        ("PRIME", "8.50%", "0.00"),
        ("VIX", "14.20", "-1.10"),
        ("HY SPREAD", "3.45%", "+0.12"),
        ("30Y MORTGAGE", "6.85%", "-0.15")
    ]
    items = []
    for name, rate, change in rates:
        color = "text-green-600" if change.startswith("-") and name != "VIX" else "text-red-600" if change.startswith("+") else "text-gray-500 dark:text-gray-400"
        items.append(f'<span class="font-semibold text-gray-700 dark:text-gray-300">{name}</span> <span class="font-mono">{rate}</span> <span class="text-[10px] {color}">{change}</span>')

    ticker_text = '<span class="mx-6 text-gray-300 dark:text-gray-600">|</span>'.join(items)
    # Adding a trailing separator for the seamless loop
    content_block = f'<div class="flex items-center shrink-0 pr-6">{ticker_text}<span class="mx-6 text-gray-300 dark:text-gray-600">|</span></div>'

    return f"""
    <div class="flex animate-marquee text-xs w-max">
        {content_block}{content_block}
    </div>
    """

# --- 3. DASHBOARD OVERVIEW ---

@app.get("/ui/view/home", response_class=HTMLResponse)
def view_home(request: Request, db: Session = Depends(get_db)):
    tx_fraud_sum = db.query(Transaction).filter(Transaction.is_fraud == 1).count() * 1450 # rough est
    loan_sum = db.query(Loan).count() * 15000
    fico_avg = 698 # Mock average for speed
    tx_total = db.query(Transaction).count()

    content = f"""
    <div class="mb-8">
        <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Home</h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Live metrics and global intelligence feeds.</p>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
            <div class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2"><i data-lucide="briefcase" class="w-4 h-4 text-indigo-500"></i> Capital Deployed</div>
            <div class="text-2xl font-bold text-gray-900 dark:text-white">${loan_sum/1000000:.1f}M</div>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
            <div class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2"><i data-lucide="users" class="w-4 h-4 text-blue-500"></i> Avg Portfolio FICO</div>
            <div class="text-2xl font-bold text-gray-900 dark:text-white">{fico_avg}</div>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
            <div class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2"><i data-lucide="alert-triangle" class="w-4 h-4 text-red-500"></i> Fraud Intercepted</div>
            <div class="text-2xl font-bold text-gray-900 dark:text-white">${tx_fraud_sum:,.0f}</div>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
            <div class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2"><i data-lucide="database" class="w-4 h-4 text-green-500"></i> Records Indexed</div>
            <div class="text-2xl font-bold text-gray-900 dark:text-white">{tx_total:,}</div>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col h-[400px]">
            <div class="px-5 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center gap-3 bg-gray-50/50 dark:bg-gray-800/50">
                <i data-lucide="trending-up" class="w-4 h-4 text-blue-500"></i>
                <h3 class="font-medium text-sm text-gray-900 dark:text-white">Live Market Intel</h3>
            </div>
            <div class="p-5 flex-1 overflow-y-auto" hx-get="/ui/widget/news/market" hx-trigger="load, every 60s">
                <div class="flex h-full items-center justify-center text-gray-400 dark:text-gray-500"><i data-lucide="loader-2" class="w-6 h-6 animate-spin"></i></div>
            </div>
        </div>
        
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col h-[400px]">
            <div class="px-5 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center gap-3 bg-gray-50/50 dark:bg-gray-800/50">
                <i data-lucide="landmark" class="w-4 h-4 text-indigo-500"></i>
                <h3 class="font-medium text-sm text-gray-900 dark:text-white">Macro Economy & Credit</h3>
            </div>
            <div class="p-5 flex-1 overflow-y-auto" hx-get="/ui/widget/news/credit" hx-trigger="load, every 60s">
                <div class="flex h-full items-center justify-center text-gray-400 dark:text-gray-500"><i data-lucide="loader-2" class="w-6 h-6 animate-spin"></i></div>
            </div>
        </div>
        
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col h-[400px]">
            <div class="px-5 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center gap-3 bg-gray-50/50 dark:bg-gray-800/50">
                <i data-lucide="shield-alert" class="w-4 h-4 text-red-500"></i>
                <h3 class="font-medium text-sm text-gray-900 dark:text-white">Cyber Threat Feed</h3>
            </div>
            <div class="p-5 flex-1 overflow-y-auto" hx-get="/ui/widget/news/fraud" hx-trigger="load, every 60s">
                <div class="flex h-full items-center justify-center text-gray-400 dark:text-gray-500"><i data-lucide="loader-2" class="w-6 h-6 animate-spin"></i></div>
            </div>
        </div>
    </div>
    """
    if "hx-request" in request.headers:
        return content
    return get_base_html(content, "home", "Home")

@app.get("/ui/widget/news/{topic}", response_class=HTMLResponse)
def get_news(topic: str):
    q = {"market": "stock+market+finance", "credit": "interest+rates+federal+reserve", "fraud": "cybersecurity+data+breach"}.get(topic, "finance")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    html = '<ul class="space-y-4">'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        xml_data = urllib.request.urlopen(req, timeout=5).read()
        root = ET.fromstring(xml_data)
        for item in root.findall('./channel/item')[:5]:
            t_el = item.find('title')
            l_el = item.find('link')
            p_el = item.find('pubDate')
            if t_el is None or l_el is None or p_el is None:
                continue
            import html as py_html
            title = py_html.escape(t_el.text)
            link = py_html.escape(l_el.text)
            pubDate = py_html.escape(p_el.text[:16])
            html += f'<li><a href="{link}" target="_blank" class="block group"><p class="text-[13px] font-medium text-gray-800 dark:text-gray-200 group-hover:text-blue-600 dark:group-hover:text-blue-400 leading-snug">{title}</p><p class="text-[10px] text-gray-400 dark:text-gray-500 mt-1">{pubDate}</p></a></li>'
    except Exception:
        html += '<li class="text-xs text-red-500">Feed unavailable</li>'
    return html + '</ul>'

# --- 4. DATA INFRASTRUCTURE (Explorer, Generator, Importer) ---

def render_tx_row(t, edit_mode=False):
    if edit_mode:
        return f"""
        <tr id="tx-row-{t.txn_id}" class="bg-blue-50/50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-800/50">
            <td class="py-2.5 px-4 text-xs font-mono text-gray-500 dark:text-gray-400">TXN-{t.txn_id:05d}</td>
            <td class="py-2 px-4"><input type="number" step="0.01" name="amount" value="{t.amount}" class="w-24 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white rounded focus:ring-1 focus:ring-blue-500 focus:border-blue-500 outline-none"></td>
            <td class="py-2 px-4"><input type="text" name="category" value="{__import__('html').escape(str(t.category))}" class="w-24 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white rounded focus:ring-1 focus:ring-blue-500 outline-none"></td>
            <td class="py-2 px-4 text-xs text-gray-500 dark:text-gray-400">
                <input type="number" name="day" value="{t.day}" class="w-12 px-1 py-1 text-xs border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white rounded inline">
                <input type="number" name="hour" value="{t.hour}" class="w-12 px-1 py-1 text-xs border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white rounded inline">
            </td>
            <td class="py-2 px-4">
                <select name="is_fraud" class="w-24 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white rounded focus:ring-1 focus:ring-blue-500 outline-none">
                    <option value="1" {'selected' if t.is_fraud else ''}>FRAUD</option>
                    <option value="0" {'selected' if not t.is_fraud else ''}>CLEAN</option>
                </select>
            </td>
            <td class="py-2 px-4 text-right whitespace-nowrap">
                <button hx-post="/ui/api/db/transaction/{t.txn_id}" hx-include="closest tr" hx-target="#tx-row-{t.txn_id}" hx-swap="outerHTML" class="text-[10px] font-bold text-white bg-blue-600 hover:bg-blue-700 px-2 py-1 rounded shadow-sm mr-1">SAVE</button>
                <button hx-get="/ui/widget/db/row/{t.txn_id}" hx-target="#tx-row-{t.txn_id}" hx-swap="outerHTML" class="text-[10px] font-bold text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:text-white bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 px-2 py-1 rounded border border-gray-200 dark:border-gray-700">CANCEL</button>
            </td>
        </tr>"""

    fraud_badge = '<span class="px-2 py-0.5 bg-red-100 dark:bg-red-900/40 text-red-700 rounded text-[10px] font-bold border border-red-200 dark:border-red-800/50">FRAUD</span>' if t.is_fraud else '<span class="px-2 py-0.5 bg-green-100 dark:bg-green-900/40 text-green-700 rounded text-[10px] font-bold border border-green-200 dark:border-green-800/50">CLEAN</span>'
    return f"""
        <tr id="tx-row-{t.txn_id}" class="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:bg-gray-800 transition-colors group">
            <td class="py-2.5 px-4 text-xs font-mono text-gray-500 dark:text-gray-400">TXN-{t.txn_id:05d}</td>
            <td class="py-2.5 px-4 text-sm font-medium text-gray-900 dark:text-white">${t.amount:,.2f}</td>
            <td class="py-2.5 px-4 text-sm text-gray-600 dark:text-gray-400 capitalize">{t.category}</td>
            <td class="py-2.5 px-4 text-xs text-gray-500 dark:text-gray-400">Day {t.day}, {t.hour:02d}:00</td>
            <td class="py-2.5 px-4">{fraud_badge}</td>
            <td class="py-2.5 px-4 text-right whitespace-nowrap">
                <button hx-get="/ui/widget/db/edit/{t.txn_id}" hx-target="#tx-row-{t.txn_id}" hx-swap="outerHTML" class="text-gray-400 dark:text-gray-500 hover:text-blue-600 dark:hover:text-blue-400 p-1 rounded hover:bg-blue-50 dark:hover:bg-blue-900/30" title="Edit row"><i data-lucide="edit-2" class="w-3.5 h-3.5"></i></button>
                <button hx-delete="/ui/api/db/transaction/{t.txn_id}" hx-target="#tx-row-{t.txn_id}" hx-swap="outerHTML" hx-confirm="Delete TXN-{t.txn_id}?" class="text-gray-400 dark:text-gray-500 hover:text-red-600 dark:hover:text-red-400 p-1 rounded hover:bg-red-50 dark:hover:bg-red-900/30 ml-1" title="Delete row"><i data-lucide="trash-2" class="w-3.5 h-3.5"></i></button>
            </td>
        </tr>"""

@app.get("/ui/widget/db/transactions", response_class=HTMLResponse)
def get_tx_page(page: int = 1, limit: int = 100, db: Session = Depends(get_db)):
    offset = (page - 1) * limit
    total_tx = db.query(Transaction).count()
    txs = db.query(Transaction).order_by(Transaction.txn_id.desc()).limit(limit).offset(offset).all()
    rows_html = "".join([render_tx_row(t) for t in txs])

    prev_dis = "disabled" if page <= 1 else ""
    next_dis = "disabled" if (offset + limit) >= total_tx else ""

    return f"""
    <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
            <thead>
                <tr class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800 text-[11px] uppercase tracking-wider text-gray-400 dark:text-gray-500 font-semibold">
                    <th class="py-3 px-4 w-32">TXN ID</th><th class="py-3 px-4 w-32">Dollar Amount</th><th class="py-3 px-4 w-32">Category</th><th class="py-3 px-4 w-32">Time Entry</th><th class="py-3 px-4 w-24">ML Status</th><th class="py-3 px-4 w-24 text-right">Actions</th>
                </tr>
            </thead>
            <tbody id="tx-table-body">{rows_html}</tbody>
        </table>
    </div>
    <div class="px-5 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 flex flex-col md:flex-row items-center justify-between rounded-b-xl gap-4">
        <form hx-get="/ui/widget/db/transactions" hx-target="#tx-table-container" hx-swap="innerHTML" class="flex items-center gap-2">
            <input type="hidden" name="page" value="1">
            <span class="text-xs text-gray-600 dark:text-gray-400 font-medium">Rows per page:</span>
            <select name="limit" onchange="this.form.dispatchEvent(new Event('submit'))" class="text-xs border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white rounded px-2 py-1 outline-none bg-white dark:bg-gray-900 cursor-pointer">
                <option value="50" {'selected' if limit==50 else ''}>50</option>
                <option value="100" {'selected' if limit==100 else ''}>100</option>
                <option value="500" {'selected' if limit==500 else ''}>500</option>
            </select>
        </form>
        <div class="flex items-center gap-6 text-xs text-gray-600 dark:text-gray-400">
            <span class="font-medium">Showing {(offset + 1):,} to {min(offset + limit, total_tx):,} of {total_tx:,}</span>
            <div class="flex items-center gap-2">
                <button hx-get="/ui/widget/db/transactions?page={page-1}&limit={limit}" hx-target="#tx-table-container" hx-swap="innerHTML" class="p-1.5 border border-gray-300 bg-white dark:bg-gray-900 rounded hover:bg-gray-100 dark:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors" {prev_dis}><i data-lucide="chevron-left" class="w-4 h-4"></i></button>
                <button hx-get="/ui/widget/db/transactions?page={page+1}&limit={limit}" hx-target="#tx-table-container" hx-swap="innerHTML" class="p-1.5 border border-gray-300 bg-white dark:bg-gray-900 rounded hover:bg-gray-100 dark:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors" {next_dis}><i data-lucide="chevron-right" class="w-4 h-4"></i></button>
            </div>
        </div>
    </div>"""

@app.get("/ui/widget/db/row/{txn_id}", response_class=HTMLResponse)
def get_tx_row(txn_id: int, db: Session = Depends(get_db)):
    t = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    return render_tx_row(t, edit_mode=False) if t else ""

@app.get("/ui/widget/db/edit/{txn_id}", response_class=HTMLResponse)
def edit_tx_row(txn_id: int, db: Session = Depends(get_db)):
    t = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    return render_tx_row(t, edit_mode=True) if t else ""

@app.post("/ui/api/db/transaction/{txn_id}", response_class=HTMLResponse)
async def update_tx_row(txn_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    t = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    if t:
        try:
            t.amount = float(form.get("amount", t.amount))
            t.category = form.get("category", t.category)
            t.day = int(form.get("day", t.day))
            t.hour = int(form.get("hour", t.hour))
            t.is_fraud = int(form.get("is_fraud", t.is_fraud))
            db.commit()
            db.refresh(t)
        except Exception:
            db.rollback()
    return render_tx_row(t, edit_mode=False) if t else ""

@app.delete("/ui/api/db/transaction/{txn_id}", response_class=HTMLResponse)
def delete_tx_row(txn_id: int, db: Session = Depends(get_db)):
    t = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    if t:
        db.delete(t)
        db.commit()
    return ""

@app.get("/ui/api/db/export/{table}")
def export_table(table: str):
    import sqlite3
    if table not in ["borrowers", "loans", "transactions"]: return "Invalid"
    conn = sqlite3.connect("sentinel.db")
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    resp = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    resp.headers["Content-Disposition"] = f"attachment; filename={table}_export.csv"
    return resp

@app.get("/ui/view/database", response_class=HTMLResponse)
def view_database(request: Request, db: Session = Depends(get_db)):
    tx_count = db.query(Transaction).count()
    loan_count = db.query(Loan).count()
    borrower_count = db.query(Borrower).count()
    db_size = os.path.getsize("sentinel.db") / (1024 * 1024) if os.path.exists("sentinel.db") else 0

    try:
        with open("sentinel.db.meta.json") as f:
            meta = json.load(f)
            meta_desc = meta.get("desc", "Unknown Dataset")
    except Exception:
        meta_desc = "Legacy Dataset"

    meta_badge = f'<div class="mt-3 flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-900 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm w-fit"><i data-lucide="database" class="w-4 h-4 text-indigo-500"></i><span>Active Engine State:</span><span class="text-gray-900 dark:text-white font-semibold">{meta_desc}</span></div>'

    content = f"""
    <div class="mb-6 flex flex-col lg:flex-row lg:justify-between lg:items-end gap-4">
        <div class="flex-1">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Risk Datamart</h2>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Read/Write access to the Sentinel SQLite data cluster.</p>
            {meta_badge}
        </div>
        <div class="flex items-center gap-2">
            <a href="/ui/api/db/export/borrowers" class="flex items-center gap-1.5 px-3 py-1.5 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:text-blue-600 hover:border-blue-200 rounded-lg text-xs font-medium shadow-sm transition-colors"><i data-lucide="download" class="w-3.5 h-3.5"></i> Borrowers</a>
            <a href="/ui/api/db/export/loans" class="flex items-center gap-1.5 px-3 py-1.5 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:text-indigo-600 hover:border-indigo-200 rounded-lg text-xs font-medium shadow-sm transition-colors"><i data-lucide="download" class="w-3.5 h-3.5"></i> Loans</a>
            <a href="/ui/api/db/export/transactions" class="flex items-center gap-1.5 px-3 py-1.5 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:text-green-600 hover:border-green-200 dark:border-green-800/50 rounded-lg text-xs font-medium shadow-sm transition-colors"><i data-lucide="download" class="w-3.5 h-3.5"></i> Transactions</a>
            <div class="flex items-center gap-2 px-3 py-1.5 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm ml-2">
                <i data-lucide="hard-drive" class="w-4 h-4 text-gray-400 dark:text-gray-500"></i>
                <span class="text-xs font-mono text-gray-600 dark:text-gray-400">sqlite:///sentinel.db ({db_size:.1f} MB)</span>
            </div>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4 shadow-sm flex items-center justify-between">
            <div><div class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Borrowers Table</div><div class="text-2xl font-bold text-gray-900 dark:text-white">{borrower_count:,} <span class="text-sm font-normal text-gray-400 dark:text-gray-500">rows</span></div></div>
            <div class="w-10 h-10 rounded-full bg-blue-50 flex items-center justify-center text-blue-600"><i data-lucide="users" class="w-5 h-5"></i></div>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4 shadow-sm flex items-center justify-between">
            <div><div class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Loans Table</div><div class="text-2xl font-bold text-gray-900 dark:text-white">{loan_count:,} <span class="text-sm font-normal text-gray-400 dark:text-gray-500">rows</span></div></div>
            <div class="w-10 h-10 rounded-full bg-indigo-50 dark:bg-indigo-900/30 flex items-center justify-center text-indigo-600"><i data-lucide="briefcase" class="w-5 h-5"></i></div>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4 shadow-sm flex items-center justify-between">
            <div><div class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Transactions Table</div><div class="text-2xl font-bold text-gray-900 dark:text-white">{tx_count:,} <span class="text-sm font-normal text-gray-400 dark:text-gray-500">rows</span></div></div>
            <div class="w-10 h-10 rounded-full bg-green-50 dark:bg-green-900/30 flex items-center justify-center text-green-600"><i data-lucide="credit-card" class="w-5 h-5"></i></div>
        </div>
    </div>
    
    <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4 shadow-sm mb-6">
        <h4 class="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3"><i data-lucide="database" class="w-4 h-4 inline-block mr-1 mb-0.5 text-indigo-500"></i>Schema Reference</h4>
        <div class="overflow-x-auto">
            <table class="w-full text-left text-sm border-collapse">
                <thead>
                    <tr class="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                        <th class="py-2 px-4 font-semibold text-gray-700 dark:text-gray-300">Table Name</th>
                        <th class="py-2 px-4 font-semibold text-gray-700 dark:text-gray-300">Primary Key</th>
                        <th class="py-2 px-4 font-semibold text-gray-700 dark:text-gray-300">Key Columns</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="border-b border-gray-100 dark:border-gray-800">
                        <td class="py-2 px-4 font-mono text-indigo-600">borrowers</td>
                        <td class="py-2 px-4 font-mono text-gray-500 dark:text-gray-400">borrower_id</td>
                        <td class="py-2 px-4 text-gray-600 dark:text-gray-400">fico_score, annual_income, emp_length_years</td>
                    </tr>
                    <tr class="border-b border-gray-100 dark:border-gray-800">
                        <td class="py-2 px-4 font-mono text-indigo-600">loans</td>
                        <td class="py-2 px-4 font-mono text-gray-500 dark:text-gray-400">loan_id</td>
                        <td class="py-2 px-4 text-gray-600 dark:text-gray-400">borrower_id (FK), loan_amount, interest_rate, is_default, ead_factor</td>
                    </tr>
                    <tr>
                        <td class="py-2 px-4 font-mono text-indigo-600">transactions</td>
                        <td class="py-2 px-4 font-mono text-gray-500 dark:text-gray-400">txn_id</td>
                        <td class="py-2 px-4 text-gray-600 dark:text-gray-400">borrower_id (FK), amount, category, is_fraud, day, hour</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl shadow-sm overflow-hidden">
        <div class="px-5 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50 flex justify-between items-center">
            <h3 class="font-medium text-sm text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="table-2" class="w-4 h-4 text-blue-500"></i> Transactions Database</h3>
        </div>
        <div id="tx-table-container">
            {get_tx_page(page=1, limit=100, db=db)}
        </div>
    </div>
    <script>lucide.createIcons();</script>
    """
    if "hx-request" in request.headers: return content
    return get_base_html(content, "database", "Data Infrastructure > Risk Datamart")

@app.get("/ui/view/generator", response_class=HTMLResponse)
def view_generator(request: Request):
    content = """
    <div class="mb-8">
        <div class="flex items-center gap-3 mb-2">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Data Forge</h2>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Synthetic Data</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Mathematically Seeded</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">SQLite</span>
            </div>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Configure macroeconomic parameters and mathematically generate a new world state.</p>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="cpu" class="w-5 h-5 text-indigo-500"></i> Engine Configuration</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">All data will be purged and reseeded deterministically.</p>
            </div>
            
            <form hx-post="/ui/api/db/generate" hx-target="#gen-status" class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Database Volume</label>
                        <select name="volume" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="small">Small (5k)</option><option value="medium" selected>Medium (20k)</option><option value="large">Large (50k)</option></select>
                        <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-1">Total row count generated</p>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Demographic Skew</label>
                        <select name="skew" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="subprime">Subprime Portfolio</option><option value="normal" selected>Normal Baseline</option><option value="superprime">Super-Prime</option></select>
                        <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-1">Shifts FICO &amp; income distributions</p>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Interest Rates</label>
                        <select name="rates" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="zirp">ZIRP (0-2%)</option><option value="normal" selected>Normal (4-6%)</option><option value="high">High Inflation</option></select>
                        <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-1">Impacts borrowing cost &amp; default rate</p>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Macro Stress</label>
                        <select name="stress" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="low">Boom (0.5x PD)</option><option value="normal" selected>Baseline (1.0x PD)</option><option value="severe">Crash (3.0x PD)</option></select>
                        <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-1">Multiplier on Probability of Default</p>
                    </div>
                    <div class="col-span-2">
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Target Fraud Rate</label>
                        <select name="fraud_rate" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="0.001">0.1% (Low)</option><option value="0.002" selected>0.2% (Standard)</option><option value="0.01">1.0% (Under Attack)</option></select>
                        <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-1">Ratio of fraudulent to legitimate transactions</p>
                    </div>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-4">
                    <i data-lucide="database-zap" class="w-4 h-4"></i> Generate Universe
                </button>
            </form>
        </div>
        
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col justify-center items-center text-center">
            <div id="gen-status" class="w-full h-full min-h-[200px] flex flex-col justify-center">
                <div class="text-gray-400 dark:text-gray-500 font-medium text-sm flex flex-col items-center justify-center">
                    <i data-lucide="database" class="w-8 h-8 mb-2 opacity-50"></i>
                    Awaiting Generation Parameters...
                </div>
            </div>
        </div>
    </div>
    <script>lucide.createIcons();</script>
    """
    if "hx-request" in request.headers: return content
    return get_base_html(content, "generator", "Data Infrastructure > Data Forge")

@app.post("/ui/api/db/generate", response_class=HTMLResponse)
async def generate_dataset(request: Request):
    form = await request.form()
    volume = form.get("volume", "medium")
    skew = form.get("skew", "normal")
    rates = form.get("rates", "normal")
    stress_str = form.get("stress", "normal")
    fraud_rate = form.get("fraud_rate", "0.002")

    stress_map = {"low": 0.5, "normal": 1.0, "severe": 3.0}
    stress_val = stress_map.get(stress_str, 1.0)

    vol_map = {"small": (1000, 1500, 5000), "medium": (5000, 6000, 20000), "large": (15000, 18000, 50000)}
    nb, nl, nt = vol_map.get(volume, vol_map["medium"])

    f_type = "synthetic"

    try:
        import json
        import subprocess
        import sys
        subprocess.run([sys.executable, "scripts/seed_db.py", str(nb), str(nl), str(nt), str(fraud_rate), str(stress_val), str(skew), str(rates), str(f_type)], check=True)
        desc = f"{volume.capitalize()} Vol | {skew.capitalize()} Skew | {rates.capitalize()} Rates | {stress_str.capitalize()} Stress | {float(fraud_rate)*100}% Fraud"
        with open("sentinel.db.meta.json", "w") as f:
            json.dump({"type": "synthetic", "desc": desc}, f)
        return '<div class="px-4 py-3 bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-200 text-indigo-800 dark:text-indigo-200 text-sm font-medium rounded-lg shadow-sm">Successfully generated! <button onclick="window.location.reload()" class="underline font-bold ml-2">Reload</button></div>'
    except Exception:
        return '<div class="px-3 py-2 bg-red-50 dark:bg-red-900/30 text-red-700 text-xs rounded-lg shadow-sm">Error: An internal error occurred. Please check server logs.</div>'

@app.post("/ui/api/db/upload", response_class=HTMLResponse)
async def upload_dataset(target_table: str = Form(...), file: UploadFile = File(...)):
    import sqlite3
    try:
        df = pd.read_csv(file.file)
        if target_table == "loans":
            if "loan_amnt" in df.columns: df.rename(columns={"loan_amnt": "loan_amount"}, inplace=True)
            if "int_rate" in df.columns: df.rename(columns={"int_rate": "interest_rate"}, inplace=True)
        elif target_table == "transactions":
            if "Class" in df.columns: df.rename(columns={"Class": "is_fraud"}, inplace=True)
            if "Amount" in df.columns: df.rename(columns={"Amount": "amount"}, inplace=True)

        if target_table not in ["borrowers", "loans", "transactions"]:
            return '<div class="px-3 py-2 bg-red-50 dark:bg-red-900/30 text-red-700 text-xs rounded-lg">Invalid table selected</div>'

        conn = sqlite3.connect("sentinel.db")
        existing_cols = pd.read_sql(f"PRAGMA table_info({target_table})", conn)['name'].tolist()
        cols_to_keep = [c for c in df.columns if c in existing_cols]
        if not cols_to_keep: return '<div class="px-3 py-2 bg-red-50 dark:bg-red-900/30 text-red-700 text-xs rounded-lg">Schema mismatch</div>'

        df_filtered = df[cols_to_keep]
        rows = len(df_filtered)
        df_filtered.to_sql(target_table, conn, if_exists="append", index=False)
        conn.close()

        try:
            with open("sentinel.db.meta.json") as f: current = json.load(f)["desc"]
            desc = current + f" + {rows} uploaded rows"
        except Exception: desc = f"Imported CSV into {target_table}"
        with open("sentinel.db.meta.json", "w") as f: json.dump({"type": "mixed", "desc": desc}, f)

        return f'<div class="px-4 py-3 bg-green-50 dark:bg-green-900/30 text-green-700 text-sm font-medium rounded-lg shadow-sm">Successfully ingested {rows} rows. <button onclick="window.location.reload()" class="underline font-bold ml-2">Reload</button></div>'
    except Exception:
        return '<div class="px-3 py-2 bg-red-50 dark:bg-red-900/30 text-red-700 text-xs rounded-lg">Error: An internal error occurred. Please check server logs.</div>'



# ==========================================
# FRAUD INTELLIGENCE ENGINE
# ==========================================
@app.get("/ui/view/fraud", response_class=HTMLResponse)
def view_fraud(request: Request):
    content = '''
    <div class="mb-8">
        <div class="flex items-center gap-3 mb-2">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Transaction Surveillance</h2>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">XGBoost Classifier</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Imbalanced Learning</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Basel AML</span>
            </div>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">XGBoost real-time transaction scoring and behavioral deviation analysis.</p>
        <div class="mt-4 flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm inline-flex">
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Raw Transactions</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Feature Engineering</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded border border-indigo-100 dark:border-indigo-800/50">XGBoost Scorer</span>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="cpu" class="w-5 h-5 text-indigo-500"></i> Model Configuration</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Tune hyperparameters to adjust model sensitivity to False Positives vs Recall.</p>
            </div>
            
            <form hx-post="/ui/api/engine/train_fraud" hx-target="#fraud-results" hx-indicator="#fraud-spinner" class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Max Tree Depth</label>
                        <select name="max_depth" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="3">3 (Shallow)</option><option value="4" selected>4 (Standard)</option><option value="6">6 (Deep)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Estimators (Trees)</label>
                        <select name="n_estimators" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="50" selected>50 Trees</option><option value="100">100 Trees</option><option value="200">200 Trees</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Learning Rate (η)</label>
                        <select name="learning_rate" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="0.01">0.01 (Conservative)</option><option value="0.1" selected>0.10 (Standard)</option><option value="0.3">0.30 (Aggressive)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Train Split</label>
                        <select name="test_size" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="0.1">90/10 Split</option><option value="0.2" selected>80/20 Split</option><option value="0.3">70/30 Split</option></select>
                    </div>
                    <div class="col-span-2">
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Classification Threshold (<span id="f-thresh-val" class="text-indigo-600">0.50</span>)</label>
                        <input type="range" name="threshold" min="0.1" max="0.9" step="0.05" value="0.5" class="w-full accent-indigo-600" oninput="document.getElementById('f-thresh-val').innerText = parseFloat(this.value).toFixed(2)">
                        <div class="flex justify-between text-[9px] text-gray-400 dark:text-gray-500 mt-0.5"><span>High Recall (catch more fraud)</span><span>High Precision (fewer false alarms)</span></div>
                    </div>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-4">
                    <i data-lucide="zap" class="w-4 h-4"></i> Execute Pipeline
                </button>
                <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">Trains on live SQLite data &middot; ~2-5s</p>
            </form>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col justify-center items-center text-center relative">
            <div id="fraud-spinner" class="htmx-indicator hidden flex-col items-center absolute inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm justify-center z-10 rounded-xl">
                <i data-lucide="loader-2" class="w-8 h-8 text-indigo-600 animate-spin mb-3"></i>
                <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Training Gradient Boosted Trees...</p>
                <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">Engineering temporal and velocity features</p>
            </div>
            <div id="fraud-results" class="w-full h-full min-h-[200px] flex flex-col justify-center">
                <div class="text-left space-y-4 w-full">
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">ROC-AUC / PR-AUC Metrics</div>
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">Feature Importance Rankings</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
        <h4 class="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3"><i data-lucide="info" class="w-4 h-4 inline-block mr-1 mb-0.5"></i>About this engine</h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Temporal Velocity</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Counts transactions per card per time window to detect spending bursts.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Behavioral Z-Score</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Measures deviation of current transaction amount from historical average.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Temporal Encoding</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Extracts cyclical patterns like hour-of-day and day-of-week risks.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Class Imbalance</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Uses scale_pos_weight to penalize missed frauds heavier than false positives.</div>
            </div>
        </div>
    </div>
    <script>lucide.createIcons();</script>
    '''
    if "hx-request" in request.headers: return content
    return get_base_html(content, "fraud", "Risk Engines > Transaction Surveillance")

@app.post("/ui/api/engine/train_fraud", response_class=HTMLResponse)
async def train_fraud(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    max_depth = int(form.get("max_depth", 4))
    n_estimators = int(form.get("n_estimators", 50))
    threshold = float(form.get("threshold", 0.5))
    learning_rate = float(form.get("learning_rate", 0.1))
    test_size = float(form.get("test_size", 0.2))

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from sentinel.fraud.detector import FraudDetector
    from sentinel.fraud.features import (
        engineer_fraud_features,
        get_fraud_feature_columns,
    )

    txns = pd.read_sql("SELECT * FROM transactions", engine.connect())
    if len(txns) < 50: return "<div class='text-red-500 font-bold'>Insufficient data. Seed database first.</div>"
    if "borrower_id" in txns.columns: txns = txns.rename(columns={"borrower_id": "card_id"})

    try:
        featured = engineer_fraud_features(txns)
        X = featured[get_fraud_feature_columns()]
        y = featured["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        model = FraudDetector(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X_train, y_train)

        metrics = model.evaluate(X_test, y_test, threshold=threshold)
        imp = model.get_feature_importance()

        imp_html = "".join([f'<div class="flex justify-between text-xs mb-1"><span class="text-gray-600 dark:text-gray-400">{name}</span><span class="font-bold">{score:.4f}</span></div>' for name, score in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]])

        return f'''
        <div class="text-left w-full h-full animate-in fade-in zoom-in duration-300">
            <h4 class="font-bold text-gray-900 dark:text-white mb-2">Test Set Performance (n={len(X_test):,})</h4>
            <div class="grid grid-cols-2 gap-3 mb-4">
                <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">ROC-AUC</div><div class="font-mono text-lg text-indigo-700 dark:text-indigo-300">{metrics.roc_auc:.4f}</div></div>
                <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">PR-AUC</div><div class="font-mono text-lg text-indigo-700 dark:text-indigo-300">{metrics.pr_auc:.4f}</div></div>
                <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">Precision (@{threshold:.2f})</div><div class="font-mono text-lg text-green-700">{metrics.precision_at_threshold:.1%}</div></div>
                <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">Recall (@{threshold:.2f})</div><div class="font-mono text-lg text-red-700">{metrics.recall_at_threshold:.1%}</div></div>
            </div>
            <h4 class="font-bold text-xs text-gray-900 dark:text-white mb-2 uppercase tracking-wider">Top Importance</h4>
            <div class="bg-blue-50/50 p-2 rounded border border-blue-100 font-mono">{imp_html}</div>
        </div>
        <script>lucide.createIcons();</script>
        '''
    except Exception: return "<div class='text-red-500 text-xs font-bold text-left'>An internal error occurred. Please check server logs.</div>"

# ==========================================
# CREDIT UNDERWRITING ENGINE
# ==========================================
@app.get("/ui/view/credit", response_class=HTMLResponse)
def view_credit(request: Request):
    content = '''
    <div class="mb-8">
        <div class="flex items-center gap-3 mb-2">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Loan Default Engine</h2>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Basel II IRB</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">PD / LGD / EAD</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Expected Loss</span>
            </div>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">XGBoost Probability of Default (PD) and Expected Loss (EL) calculations.</p>
        <div class="mt-4 flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm inline-flex">
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Borrower Data</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">FICO + DTI Features</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">XGBoost PD</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded border border-indigo-100 dark:border-indigo-800/50">EL = PD &times; LGD &times; EAD</span>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="briefcase" class="w-5 h-5 text-indigo-500"></i> Model Configuration</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Tune the loan evaluation parameters to assess portfolio default risk.</p>
            </div>
            
            <form hx-post="/ui/api/engine/train_credit" hx-target="#credit-results" hx-indicator="#credit-spinner" class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">PD Max Tree Depth</label>
                        <select name="max_depth" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="2">2 (Very Shallow)</option><option value="3" selected>3 (Standard)</option><option value="5">5 (Deep)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">PD Estimators (Trees)</label>
                        <select name="n_estimators" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="50" selected>50 Trees</option><option value="100">100 Trees</option><option value="200">200 Trees</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Learning Rate (η)</label>
                        <select name="learning_rate" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="0.01">0.01 (Conservative)</option><option value="0.1" selected>0.10 (Standard)</option><option value="0.3">0.30 (Aggressive)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">LGD / EAD Depth</label>
                        <select name="lgd_ead_depth" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="2" selected>2 (Shallow)</option><option value="3">3 (Standard)</option><option value="4">4 (Deep)</option></select>
                    </div>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-4">
                    <i data-lucide="zap" class="w-4 h-4"></i> Execute Pipeline
                </button>
                <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">Trains on live SQLite data &middot; ~2-5s</p>
            </form>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col justify-center items-center text-center relative">
            <div id="credit-spinner" class="htmx-indicator hidden flex-col items-center absolute inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm justify-center z-10 rounded-xl">
                <i data-lucide="loader-2" class="w-8 h-8 text-indigo-600 animate-spin mb-3"></i>
                <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Training Credit Risk Models...</p>
                <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">Calculating PD, LGD, and EAD</p>
            </div>
            <div id="credit-results" class="w-full h-full min-h-[200px] flex flex-col justify-center">
                <div class="text-left space-y-4 w-full">
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">Expected Loss (EL) Estimates</div>
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">Probability of Default (PD) Profile</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
        <h4 class="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3"><i data-lucide="info" class="w-4 h-4 inline-block mr-1 mb-0.5"></i>About this engine</h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Probability of Default (PD)</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Likelihood borrower defaults within 1 year, output by XGBoost classifier.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Loss Given Default (LGD)</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Percentage of exposure lost if default occurs. Reduced by collateral.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Exposure at Default (EAD)</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Estimated outstanding balance at the time of default.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Expected Loss formula</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">EL = PD &times; LGD &times; EAD. The standard Basel framework for credit risk capital.</div>
            </div>
        </div>
    </div>
    <script>lucide.createIcons();</script>
    '''
    if "hx-request" in request.headers: return content
    return get_base_html(content, "credit", "Risk Engines > Loan Default Engine")

@app.post("/ui/api/engine/train_credit", response_class=HTMLResponse)
async def train_credit(request: Request):
    form = await request.form()
    max_depth = int(form.get("max_depth", 3))
    n_estimators = int(form.get("n_estimators", 50))
    learning_rate = float(form.get("learning_rate", 0.1))
    lgd_ead_depth = int(form.get("lgd_ead_depth", 2))

    import pandas as pd
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    from sentinel.credit.models import (
        EADModel,
        ExpectedLossEngine,
        LGDModel,
        XGBoostPDModel,
    )

    try:
        loans = pd.read_sql("SELECT * FROM loans JOIN borrowers ON loans.borrower_id = borrowers.borrower_id", engine.connect())
        if len(loans) < 50: return "<div class='text-red-500 font-bold'>Insufficient data. Seed DB.</div>"

        features = ["fico_score", "annual_income", "emp_length_years", "loan_amount", "interest_rate", "dti", "is_secured"]
        X = loans[features]
        y = loans["is_default"]

        # 1. Train PD Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pd_model = XGBoostPDModel(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
        pd_model.fit(X_train, y_train)
        preds = pd_model.predict_proba(X_test)

        auc = roc_auc_score(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        imp = pd_model.get_feature_importance()

        # 2. Train LGD & EAD Models on defaults — depth controlled by user
        defaults = loans[loans["is_default"] == 1]
        total_el = 0.0
        if len(defaults) > 10:
            lgd_model = LGDModel(max_depth=lgd_ead_depth)
            lgd_model.fit(defaults[features], defaults["lgd"])

            ead_model = EADModel(max_depth=lgd_ead_depth)
            ead_model.fit(defaults[features], defaults["ead_factor"])

            # 3. Calculate Expected Loss for the test set
            loss_engine = ExpectedLossEngine(pd_model, lgd_model, ead_model)
            el_df = loss_engine.predict_expected_loss(X_test, loans.loc[X_test.index, "loan_amount"])
            total_el = el_df["Expected_Loss"].sum()

        imp_html = "".join([f'<div class="flex justify-between text-[10px] mb-1"><span class="text-gray-600 dark:text-gray-400">{name}</span><span class="font-bold">{score:.4f}</span></div>' for name, score in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]])

        return f"""
        <div class="text-left w-full h-full animate-in fade-in zoom-in duration-300 flex flex-col justify-between">
            <div>
                <h4 class="font-bold text-gray-900 dark:text-white mb-2">Test Set Performance (n={len(X_test):,})</h4>
                <div class="grid grid-cols-2 gap-3 mb-4">
                    <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">ROC-AUC</div><div class="font-mono text-lg text-indigo-700 dark:text-indigo-300">{auc:.4f}</div></div>
                    <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">Brier Score</div><div class="font-mono text-lg text-green-700">{brier:.4f}</div></div>
                    <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700 col-span-2 flex justify-between items-center"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">Est. Portfolio Expected Loss</div><div class="font-mono text-xl text-red-600 font-bold">${total_el:,.0f}</div></div>
                </div>
            </div>
            <div>
                <h4 class="font-bold text-[10px] text-gray-900 dark:text-white mb-1 uppercase tracking-wider">Top Importance (PD)</h4>
                <div class="bg-blue-50/50 p-2 rounded border border-blue-100 font-mono">{imp_html}</div>
            </div>
        </div>
        <script>lucide.createIcons();</script>
        """
    except Exception: return "<div class='text-red-500 text-xs font-bold text-left'>An internal error occurred. Please check server logs.</div>"

# ==========================================
# MARKET RISK ANALYTICS ENGINE
# ==========================================
@app.get("/ui/view/market", response_class=HTMLResponse)
def view_market(request: Request):
    content = '''
    <div class="mb-8">
        <div class="flex items-center gap-3 mb-2">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Portfolio VaR</h2>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Basel III</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Geometric Brownian Motion</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Kupiec POF</span>
            </div>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Parametric and historical Value at Risk (VaR) with Expected Shortfall (ES).</p>
        <div class="mt-4 flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm inline-flex">
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Market Prices</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Returns</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Rolling Covariance</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded border border-indigo-100 dark:border-indigo-800/50">VaR + ES</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Backtesting</span>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="line-chart" class="w-5 h-5 text-indigo-500"></i> Engine Configuration</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Configure confidence intervals and Monte Carlo simulation parameters.</p>
            </div>
            
            <form hx-post="/ui/api/engine/train_market" hx-target="#market-results" hx-indicator="#market-spinner" class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Simulation Method</label>
                        <select name="method" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="parametric">Delta-Normal (Parametric)</option><option value="historical">Historical Simulation</option><option value="monte_carlo" selected>Monte Carlo (GBM)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Confidence Level</label>
                        <select name="confidence_level" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="0.90">90% VaR</option><option value="0.95" selected>95% VaR</option><option value="0.99">99% VaR (Extreme)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Rolling Window</label>
                        <select name="window_size" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="126">126 Days (6m)</option><option value="252" selected>252 Days (1yr)</option><option value="504">504 Days (2yr)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Portfolio Value ($)</label>
                        <select name="portfolio_value" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="100000">$100,000</option><option value="1000000" selected>$1,000,000</option><option value="10000000">$10,000,000</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">MC Simulations</label>
                        <select name="n_simulations" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="1000">1,000 paths</option><option value="5000" selected>5,000 paths</option><option value="10000">10,000 paths</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">VaR Horizon</label>
                        <select name="time_horizon" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="1" selected>1-Day VaR (Standard)</option><option value="10">10-Day VaR (Basel III)</option><option value="21">21-Day VaR (Monthly)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Random Seed</label>
                        <select name="random_seed" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="42" selected>42 (Reproducible)</option><option value="0">0 (Random)</option><option value="2024">2024 (Benchmark)</option></select>
                    </div>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-4">
                    <i data-lucide="zap" class="w-4 h-4"></i> Execute Pipeline
                </button>
                <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">Generates synthetic price histories &middot; ~1-3s</p>
            </form>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col justify-center items-center text-center relative">
            <div id="market-spinner" class="htmx-indicator hidden flex-col items-center absolute inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm justify-center z-10 rounded-xl">
                <i data-lucide="loader-2" class="w-8 h-8 text-indigo-600 animate-spin mb-3"></i>
                <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Simulating Market Trajectories...</p>
                <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">Computing covariance matrices</p>
            </div>
            <div id="market-results" class="w-full h-full min-h-[200px] flex flex-col justify-center">
                <div class="text-left space-y-4 w-full">
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">Value at Risk (VaR)</div>
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">Expected Shortfall (ES)</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
        <h4 class="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3"><i data-lucide="info" class="w-4 h-4 inline-block mr-1 mb-0.5"></i>About this engine</h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">VaR Definition</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Maximum expected loss over a specific time horizon at a given confidence level.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Expected Shortfall</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Average of all losses that exceed the VaR threshold. Captures tail risk.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Kupiec Test</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Likelihood ratio test for VaR model accuracy by analyzing backtest exception frequencies.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Rolling Window Bias</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Historical simulation ghost effects caused by large drops exiting the calculation window.</div>
            </div>
        </div>
    </div>
    <script>lucide.createIcons();</script>
    '''
    if "hx-request" in request.headers: return content
    return get_base_html(content, "market", "Risk Engines > Portfolio VaR")

@app.post("/ui/api/engine/train_market", response_class=HTMLResponse)
async def train_market(request: Request):
    form = await request.form()
    method = form.get("method", "monte_carlo")
    confidence_level = float(form.get("confidence_level", 0.95))
    window_size = int(form.get("window_size", 252))
    portfolio_value = float(form.get("portfolio_value", 1_000_000))
    _n_simulations = int(form.get("n_simulations", 5000))  # reserved for future MC engine
    time_horizon = int(form.get("time_horizon", 1))
    random_seed = int(form.get("random_seed", 42))

    try:
        import numpy as np

        from sentinel.data.market import forward_fill_prices, load_prices
        from sentinel.quant.portfolio import PortfolioDefinition
        from sentinel.quant.risk_measures import compute_risk_measures
        from sentinel.quant.rolling_backtest import run_rolling_backtest

        prices = forward_fill_prices(load_prices())
        portfolio = PortfolioDefinition(weights={"SPY": 0.40, "AAPL": 0.30, "MSFT": 0.30})

        # Run rolling backtest (VaR + Kupiec)
        res = run_rolling_backtest(
            prices, portfolio, window_size=window_size,
            confidence_level=confidence_level, method=method
        )
        k = res.kupiec_result
        kupiec_color = "text-green-700" if k.is_accepted else "text-red-700"
        kupiec_text = "PASS" if k.is_accepted else "FAIL"

        # Run full VaR / ES risk measure report on actual P&L
        actual_pnl = res.daily_results["Actual_PnL"].dropna().values
        risk_report = compute_risk_measures(
            pnl=actual_pnl,
            confidence_levels=[0.90, 0.95, 0.99],
            portfolio_value=portfolio_value
        )

        # Scale VaR/ES by √time_horizon (Basel III square-root-of-time rule)
        scale = np.sqrt(time_horizon)
        rm = next((m for m in risk_report.risk_measures if abs(m.confidence_level - confidence_level) < 0.01), risk_report.risk_measures[1])
        var_scaled = rm.var_dollar * scale
        es_scaled = rm.es_dollar * scale
        horizon_label = f"{time_horizon}-Day" if time_horizon > 1 else "1-Day"

        # ES backtest result
        es_res = res.es_result
        es_html = ""
        if es_res:
            es_color = "text-green-700" if es_res.is_accepted else "text-red-700"
            es_html = f'<div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">ES Backtest</div><div class="font-mono text-base {es_color} font-bold">{"PASS" if es_res.is_accepted else "FAIL"} (p={es_res.p_value:.3f})</div></div>'

        skew_color = "text-red-600" if risk_report.skewness < -0.5 else "text-green-600"

        return f"""
        <div class="text-left w-full animate-in fade-in zoom-in duration-300">
            <h4 class="font-bold text-gray-900 dark:text-white mb-3">Results: {res.method_name.capitalize()} @ {confidence_level*100:.0f}% | {horizon_label} | ${portfolio_value:,.0f}</h4>
            <div class="grid grid-cols-2 gap-2 mb-3">
                <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">VaR ({confidence_level*100:.0f}%) {horizon_label}</div><div class="font-mono text-base text-red-600 font-bold">${var_scaled:,.0f}</div><div class="text-[9px] text-gray-400 dark:text-gray-500">{rm.var:.4f} daily × √{time_horizon}</div></div>
                <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">ES / CVaR {horizon_label}</div><div class="font-mono text-base text-red-700 font-bold">${es_scaled:,.0f}</div><div class="text-[9px] text-gray-400 dark:text-gray-500">{rm.es:.4f} avg tail × √{time_horizon}</div></div>
                <div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">Kupiec POF</div><div class="font-mono text-base {kupiec_color} font-bold">{kupiec_text}</div><div class="text-[9px] text-gray-400 dark:text-gray-500">p={k.p_value:.4f} | {k.actual_breaches}/{k.expected_breaches:.1f} breaches</div></div>
                {es_html if es_html else f'<div class="bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">Worst Loss Day</div><div class="font-mono text-base text-red-600">${risk_report.worst_loss*portfolio_value:,.0f}</div></div>'}
            </div>
            <div class="mt-3 bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 text-xs">
                <p class="font-bold text-gray-700 dark:text-gray-300 mb-1">Interpretation & Assumptions:</p>
                <p class="text-gray-500 dark:text-gray-400">There is a <strong>{100 - confidence_level*100:.1f}% probability</strong> that the portfolio will lose more than <strong>${var_scaled:,.0f}</strong> over the next <strong>{time_horizon} trading day(s)</strong>, under the {res.method_name} distribution assumptions using random seed {random_seed}. The Expected Shortfall estimates that <em>if</em> this tail event occurs, the average loss would be <strong>${es_scaled:,.0f}</strong>.</p>
            </div>
            <div class="grid grid-cols-4 gap-2 mt-3">
                <div class="bg-blue-50/50 p-2 rounded border border-blue-100 text-center"><div class="text-[9px] text-blue-700 font-bold uppercase">Skewness</div><div class="font-mono text-sm {skew_color}">{risk_report.skewness:.3f}</div></div>
                <div class="bg-blue-50/50 p-2 rounded border border-blue-100 text-center"><div class="text-[9px] text-blue-700 font-bold uppercase">Kurtosis</div><div class="font-mono text-sm text-gray-700 dark:text-gray-300">{risk_report.kurtosis:.3f}</div></div>
                <div class="bg-blue-50/50 p-2 rounded border border-blue-100 text-center"><div class="text-[9px] text-blue-700 font-bold uppercase">Days Tested</div><div class="font-mono text-sm text-gray-700 dark:text-gray-300">{k.observations}</div></div>
                <div class="bg-blue-50/50 p-2 rounded border border-blue-100 text-center"><div class="text-[9px] text-blue-700 font-bold uppercase">Seed</div><div class="font-mono text-sm text-gray-700 dark:text-gray-300">{random_seed}</div></div>
            </div>
        </div>
        <script>lucide.createIcons();</script>
        """
    except Exception:
        return "<div class='text-red-500 text-xs font-bold text-left p-4 bg-red-50 dark:bg-red-900/30 rounded'>An internal error occurred. Please check server logs.</div>"

# ==========================================
# STRESS TESTING ENGINE
# ==========================================
@app.get("/ui/view/stress", response_class=HTMLResponse)
def view_stress(request: Request):
    content = '''
    <div class="mb-8">
        <div class="flex items-center gap-3 mb-2">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Scenario Analysis</h2>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">CCAR</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Basel Pillar 2</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Hypothetical + Historical</span>
            </div>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Stress testing and portfolio revaluation under severe macro-economic shocks.</p>
        <div class="mt-4 flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm inline-flex">
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Portfolio Weights</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Shock Multipliers</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded border border-indigo-100 dark:border-indigo-800/50">Instantaneous P&amp;L Impact</span>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="flame" class="w-5 h-5 text-indigo-500"></i> Engine Configuration</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Select historical scenarios or apply custom parallel shifts.</p>
            </div>
            
            <form hx-post="/ui/api/engine/stress_test" hx-target="#stress-results" hx-indicator="#stress-spinner" class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div class="col-span-2">
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Scenario Type</label>
                        <select name="scenario_type" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="hypothetical" selected>Hypothetical (Custom Shocks)</option><option value="historical">Historical Crisis Replay</option></select>
                    </div>
                    <div class="col-span-2">
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Preset (Hypothetical)</label>
                        <select name="hypo_preset" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="tech_crash" selected>Tech Crash (−30% AAPL/MSFT)</option><option value="market_rout">Market Rout (−20% SPY)</option><option value="flight_to_safety">Flight to Safety (−10% all)</option></select>
                    </div>
                    <div class="col-span-2">
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Historical Crisis</label>
                        <select name="hist_preset" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="covid" selected>2022 Q1 Selloff (Jan–Mar 2022)</option><option value="gfc">2022 Rate Shock (Apr–Oct 2022)</option><option value="dotcom">2022 Bear Market Bottom (Aug–Dec 2022)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Portfolio Value ($)</label>
                        <select name="portfolio_value" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="100000">$100,000</option><option value="1000000" selected>$1,000,000</option><option value="10000000">$10,000,000</option></select>
                    </div>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-4">
                    <i data-lucide="zap" class="w-4 h-4"></i> Execute Pipeline
                </button>
                <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">Applies shocks to current holdings &middot; ~1s</p>
            </form>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col justify-center items-center text-center relative">
            <div id="stress-spinner" class="htmx-indicator hidden flex-col items-center absolute inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm justify-center z-10 rounded-xl">
                <i data-lucide="loader-2" class="w-8 h-8 text-indigo-600 animate-spin mb-3"></i>
                <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Revaluing Portfolio...</p>
                <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">Applying instantaneous shocks</p>
            </div>
            <div id="stress-results" class="w-full h-full min-h-[200px] flex flex-col justify-center">
                <div class="text-left space-y-4 w-full">
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">P&amp;L Impact by Asset Class</div>
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">Worst-Case Drawdown Metrics</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
        <h4 class="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3"><i data-lucide="info" class="w-4 h-4 inline-block mr-1 mb-0.5"></i>About this engine</h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Hypothetical Shocks</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Instantaneous parallel shift applied to portfolio weights (e.g. Equities -20%, Rates +100bps).</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Historical Replay</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Methodology applying exact historical market movements to today's exposures.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Max Drawdown</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">The maximum observed loss from a peak to a trough of a portfolio before a new peak is attained.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Peak-to-Trough</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Measures the full magnitude of a market crash across the entire stressed period.</div>
            </div>
        </div>
    </div>
    <script>lucide.createIcons();</script>
    '''
    if "hx-request" in request.headers: return content
    return get_base_html(content, "stress", "Risk Engines > Scenario Analysis")

@app.post("/ui/api/engine/stress_test", response_class=HTMLResponse)
async def run_stress_test(request: Request):
    form = await request.form()
    scenario_type = form.get("scenario_type", "hypothetical")
    portfolio_value = float(form.get("portfolio_value", 1_000_000))

    try:
        from sentinel.data.market import forward_fill_prices, load_prices
        from sentinel.quant.portfolio import PortfolioDefinition
        from sentinel.quant.stress import (
            Scenario,
            Shock,
            apply_hypothetical_scenario,
            historical_scenario_impact,
        )

        prices = forward_fill_prices(load_prices())
        portfolio = PortfolioDefinition(weights={"SPY": 0.40, "AAPL": 0.30, "MSFT": 0.30})

        if scenario_type == "hypothetical":
            preset = form.get("hypo_preset", "tech_crash")
            presets = {
                "tech_crash": ("Tech Sector Crash", -0.10, -0.30, -0.25),
                "market_rout": ("Broad Market Rout", -0.20, -0.20, -0.20),
                "flight_to_safety": ("Flight to Safety", -0.02, -0.05, -0.05),
            }
            if preset in presets:
                name, spy_s, aapl_s, msft_s = presets[preset]
            else:
                name = "Custom Shock"
                spy_s = float(form.get("spy_shock", -10)) / 100
                aapl_s = float(form.get("aapl_shock", -30)) / 100
                msft_s = float(form.get("msft_shock", -25)) / 100

            scenario = Scenario(name=name, shocks=[
                Shock("SPY", spy_s), Shock("AAPL", aapl_s), Shock("MSFT", msft_s)
            ])
            result = apply_hypothetical_scenario(portfolio, scenario, portfolio_value)

        else:
            hist_preset = form.get("hist_preset", "covid")
            crisis_map = {
                "covid":  ("COVID Rebound Selloff",    "2022-01-03", "2022-03-14"),
                "gfc":    ("2022 Rate Shock (Fed)",    "2022-04-01", "2022-10-14"),
                "dotcom": ("2022 Bear Market Bottom",  "2022-08-15", "2022-12-30"),
            }
            name, start, end = crisis_map.get(hist_preset, crisis_map["covid"])
            result = historical_scenario_impact(prices, portfolio, start, end, name, portfolio_value)

        pnl_color = "text-green-600" if result.portfolio_pnl_dollar > 0 else "text-red-600"
        asset_rows = "".join([
            f'<div class="flex justify-between items-center py-1 border-b border-gray-100 dark:border-gray-800 last:border-0"><span class="text-sm text-gray-700 dark:text-gray-300 font-medium">{asset}</span><span class="font-mono text-sm {"text-red-600" if pnl < 0 else "text-green-600"} font-bold">{pnl*100:.2f}%</span></div>'
            for asset, pnl in result.asset_pnls_pct.items()
        ])

        return f"""
        <div class="text-left w-full animate-in fade-in zoom-in duration-300">
            <h4 class="font-bold text-gray-900 dark:text-white mb-1">{result.scenario_name}</h4>
            <p class="text-xs text-gray-500 dark:text-gray-400 mb-4">Instantaneous portfolio impact on $1,000,000 notional.</p>
            <div class="grid grid-cols-2 gap-3 mb-4">
                <div class="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 col-span-2">
                    <div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase mb-1">Portfolio P&L</div>
                    <div class="font-mono text-3xl {pnl_color} font-black">${result.portfolio_pnl_dollar:,.0f}</div>
                    <div class="text-xs text-gray-400 dark:text-gray-500 mt-1">{result.portfolio_pnl_pct*100:.2f}% return impact</div>
                </div>
            </div>
            <h4 class="font-bold text-[10px] text-gray-900 dark:text-white mb-2 uppercase tracking-wider">Per-Asset Impact</h4>
            <div class="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700">{asset_rows}</div>
        </div>
        <script>lucide.createIcons();</script>
        """
    except Exception:
        import traceback
        return f"<div class='text-red-500 text-xs font-bold p-4 bg-red-50 dark:bg-red-900/30 rounded'><pre>{traceback.format_exc()}</pre></div>"

# ==========================================
# VOLATILITY LAB
# ==========================================
@app.get("/ui/view/volatility", response_class=HTMLResponse)
def view_volatility(request: Request):
    content = '''
    <div class="mb-8">
        <div class="flex items-center gap-3 mb-2">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Volatility Surface</h2>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">GARCH(1,1)</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">RiskMetrics EWMA</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Annualized &sigma;</span>
            </div>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Advanced econometric modeling of conditional variance and volatility clustering.</p>
        <div class="mt-4 flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm inline-flex">
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Log Returns</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Rolling Std / EWMA / GARCH MLE</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded border border-indigo-100 dark:border-indigo-800/50">Annualized Volatility</span>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="activity" class="w-5 h-5 text-indigo-500"></i> Model Configuration</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Select time horizons and models for variance estimation.</p>
            </div>
            
            <form hx-post="/ui/api/engine/volatility" hx-target="#vol-results" hx-indicator="#vol-spinner" class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Asset</label>
                        <select name="asset" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="SPY" selected>SPY (S&amp;P 500)</option><option value="AAPL">AAPL</option><option value="MSFT">MSFT</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Rolling Window</label>
                        <select name="window" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="21">21 Days (1m)</option><option value="63" selected>63 Days (3m)</option><option value="252">252 Days (1yr)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">EWMA Lambda (λ)</label>
                        <select name="ewma_lambda" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="0.94" selected>0.94 (RiskMetrics)</option><option value="0.97">0.97 (Slow Decay)</option><option value="0.90">0.90 (Fast Decay)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">GARCH Error Distribution</label>
                        <select name="garch_dist" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="normal" selected>Normal (Gaussian)</option><option value="t">Student-t (Fat Tails)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">GARCH Order (q)</label>
                        <select name="garch_q" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="1" selected>q=1 — GARCH(1,1)</option><option value="2">q=2 — GARCH(1,2)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Output</label>
                        <select name="annualize" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="true" selected>Annualized (×√252)</option><option value="false">Daily Volatility</option></select>
                    </div>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-4">
                    <i data-lucide="zap" class="w-4 h-4"></i> Run Volatility Analysis
                </button>
                <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">Optimizes GARCH parameters via MLE &middot; ~2-6s</p>
            </form>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col justify-center items-center text-center relative">
            <div id="vol-spinner" class="htmx-indicator hidden flex-col items-center absolute inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm justify-center z-10 rounded-xl">
                <i data-lucide="loader-2" class="w-8 h-8 text-indigo-600 animate-spin mb-3"></i>
                <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Estimating Volatility Models...</p>
                <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">Solving MLE for GARCH</p>
            </div>
            <div id="vol-results" class="w-full h-full min-h-[200px] flex flex-col justify-center">
                <div class="text-left space-y-4 w-full">
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">GARCH(1,1) Volatility</div>
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">EWMA vs Rolling Std</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
        <h4 class="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3"><i data-lucide="info" class="w-4 h-4 inline-block mr-1 mb-0.5"></i>About this engine</h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Volatility Clustering</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">The tendency of large changes in asset prices to be followed by large changes.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">EWMA Lambda Decay</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Exponentially weights recent observations more heavily than distant ones.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">GARCH Persistence</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">&alpha; + &beta; measures how long shocks to volatility take to decay.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Long-run Variance</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">The mean-reverting baseline level of volatility in a GARCH process.</div>
            </div>
        </div>
    </div>
    <script>lucide.createIcons();</script>
    '''
    if "hx-request" in request.headers: return content
    return get_base_html(content, "volatility", "Risk Engines > Volatility Surface")

@app.post("/ui/api/engine/volatility", response_class=HTMLResponse)
async def run_volatility(request: Request):
    form = await request.form()
    asset = form.get("asset", "SPY")
    window = int(form.get("window", 63))
    ewma_lambda = float(form.get("ewma_lambda", 0.94))
    garch_dist = form.get("garch_dist", "normal")
    garch_q = int(form.get("garch_q", 1))
    annualize = form.get("annualize", "true") == "true"

    try:
        from sentinel.data.market import forward_fill_prices, load_prices
        from sentinel.quant.returns import simple_returns
        from sentinel.quant.volatility import (
            ewma_volatility,
            garch_volatility,
            rolling_volatility,
        )

        prices = forward_fill_prices(load_prices())
        if asset not in prices.columns:
            return f"<div class='text-red-500 font-bold'>Error: Asset ticker '{asset}' not found in the SQLite market database.</div>"
        returns = simple_returns(prices[[asset]])

        roll_vol = rolling_volatility(returns, window=window, annualize=annualize)
        ewma_vol = ewma_volatility(returns, lambda_=ewma_lambda, annualize=annualize)
        garch_res = garch_volatility(returns[asset], q=garch_q, dist=garch_dist)

        roll_latest = float(roll_vol[asset].dropna().iloc[-1])
        ewma_latest = float(ewma_vol[asset].dropna().iloc[-1])
        garch_latest = float(garch_res.long_run_volatility)

        omega, alpha, beta = garch_res.omega, garch_res.alpha, garch_res.beta
        persistence = garch_res.persistence
        annualize_label = "Annualized" if annualize else "Daily"


        persist_color = "text-red-600" if persistence > 0.98 else "text-yellow-600" if persistence > 0.95 else "text-green-600"

        return f"""
        <div class="text-left w-full animate-in fade-in zoom-in duration-300">
            <h4 class="font-bold text-gray-900 dark:text-white mb-3">Volatility Comparison: {asset}</h4>
            <div class="grid grid-cols-1 gap-2 mb-4">
                <div class="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 flex justify-between items-center">
                    <div><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">Rolling Std ({window}d) · {annualize_label}</div><div class="text-xs text-gray-400 dark:text-gray-500">Simple historical std deviation</div></div>
                    <div class="font-mono text-xl text-indigo-700 dark:text-indigo-300 font-bold">{roll_latest*100:.2f}%</div>
                </div>
                <div class="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 flex justify-between items-center">
                    <div><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">EWMA (λ={ewma_lambda}) · {annualize_label}</div><div class="text-xs text-gray-400 dark:text-gray-500">RiskMetrics exponentially-weighted</div></div>
                    <div class="font-mono text-xl text-blue-700 font-bold">{ewma_latest*100:.2f}%</div>
                </div>
                <div class="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 flex justify-between items-center">
                    <div><div class="text-[10px] text-gray-500 dark:text-gray-400 font-bold uppercase">GARCH(1,{garch_q}) MLE · {garch_dist.capitalize()} Errors</div><div class="text-xs text-gray-400 dark:text-gray-500">Bollerslev (1986). Long-run annualized vol.</div></div>
                    <div class="font-mono text-xl text-purple-700 font-bold">{garch_latest*100:.2f}%</div>
                </div>
            </div>
            <h4 class="font-bold text-[10px] text-gray-900 dark:text-white mb-2 uppercase tracking-wider">GARCH(1,{garch_q}) Parameters</h4>
            <div class="mt-3 bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 text-xs">
                <p class="font-bold text-gray-700 dark:text-gray-300 mb-1">Interpretation & Assumptions:</p>
                <p class="text-gray-500 dark:text-gray-400">The GARCH(1,{garch_q}) model estimates long-run annualized volatility for this asset at <strong>{garch_latest*100:.2f}%</strong>. Persistence (alpha + beta = {persistence:.4f}) measures how long volatility shocks last. Values above 0.98 indicate highly persistent volatility clustering, meaning market turbulence tends to sustain itself.</p>
            </div>
            <div class="grid grid-cols-4 gap-2 mt-3">
                <div class="bg-purple-50/50 p-2 rounded border border-purple-100 text-center"><div class="text-[9px] text-purple-700 font-bold uppercase">ω (omega)</div><div class="font-mono text-sm">{omega:.6f}</div></div>
                <div class="bg-purple-50/50 p-2 rounded border border-purple-100 text-center"><div class="text-[9px] text-purple-700 font-bold uppercase">α (shock)</div><div class="font-mono text-sm">{alpha:.4f}</div></div>
                <div class="bg-purple-50/50 p-2 rounded border border-purple-100 text-center"><div class="text-[9px] text-purple-700 font-bold uppercase">β (persist)</div><div class="font-mono text-sm {persist_color}">{beta:.4f}</div></div>
                <div class="bg-purple-50/50 p-2 rounded border border-purple-100 text-center"><div class="text-[9px] text-purple-700 font-bold uppercase">α+β</div><div class="font-mono text-sm {persist_color}">{persistence:.4f}</div></div>
            </div>
            <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-2 italic">Persistence α+β={persistence:.4f} — values >0.98 indicate highly persistent volatility clustering.</p>
        </div>
        <script>lucide.createIcons();</script>
        """
    except Exception:
        import traceback
        return f"<div class='text-red-500 text-xs font-bold p-4 bg-red-50 dark:bg-red-900/30 rounded'><pre>{traceback.format_exc()}</pre></div>"

# ==========================================
# UNIFIED CUSTOMER 360
# ==========================================
@app.get("/ui/view/customer_360", response_class=HTMLResponse)
def view_customer_360(request: Request):
    content = '''
    <div class="mb-8">
        <div class="flex items-center gap-3 mb-2">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Obligor 360</h2>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Cross-Engine</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Identity Resolution</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Unified Risk</span>
            </div>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Holistic entity resolution combining credit and fraud vectors.</p>
        <div class="mt-4 flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm inline-flex">
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Borrower ID</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">Credit Engine (EL) + Fraud Engine (Exposure)</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-400 dark:text-gray-500"></i>
            <span class="bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded border border-indigo-100 dark:border-indigo-800/50">Total Risk Score</span>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="user-check" class="w-5 h-5 text-indigo-500"></i> Entity Search</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Resolve and score a specific individual across all databases.</p>
            </div>
            
            <form hx-post="/ui/api/customer_360/search" hx-target="#c360-results" hx-indicator="#c360-spinner" class="space-y-4">
                <div>
                    <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Borrower ID</label>
                    <input type="text" name="borrower_id" placeholder="e.g. B-00001" class="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 dark:text-white rounded text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500" required>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-4">
                    <i data-lucide="search" class="w-4 h-4"></i> Query Global Index
                </button>
                <p class="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">Fuses data from 3 engines &middot; ~2s</p>
            </form>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col justify-center items-center text-center relative">
            <div id="c360-spinner" class="htmx-indicator hidden flex-col items-center absolute inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm justify-center z-10 rounded-xl">
                <i data-lucide="loader-2" class="w-8 h-8 text-indigo-600 animate-spin mb-3"></i>
                <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Resolving Identity...</p>
                <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">Cross-referencing credit and fraud databases</p>
            </div>
            <div id="c360-results" class="w-full h-full min-h-[200px] flex flex-col justify-center">
                <div class="text-left space-y-4 w-full">
                    <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-4 h-24 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">Unified Risk Assessment</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
        <h4 class="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3"><i data-lucide="info" class="w-4 h-4 inline-block mr-1 mb-0.5"></i>About this engine</h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Identity Resolution</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Cross-engine fusion of credit and fraud risk vectors for a single entity.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Credit-Fraud Fusion</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Combines long-term creditworthiness with real-time transactional anomalies.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">EL = PD &times; LGD &times; EAD</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Retrieves the Expected Loss evaluated by the Credit Underwriting Engine.</div>
            </div>
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div class="text-xs font-bold text-gray-900 dark:text-white mb-1">Fraud Velocity Scoring</div>
                <div class="text-[10px] text-gray-500 dark:text-gray-400">Aggregates flagged transactional volume from the Transaction Surveillance Engine.</div>
            </div>
        </div>
    </div>
    <script>lucide.createIcons();</script>
    '''
    if "hx-request" in request.headers: return content
    return get_base_html(content, "customer_360", "Risk Engines > Obligor 360")

@app.post("/ui/api/customer_360/search", response_class=HTMLResponse)
async def search_customer_360(request: Request):
    form = await request.form()
    raw_bid = form.get("borrower_id", "").strip()
    if not raw_bid:
        return "<div class='text-red-500 text-sm font-bold p-4 bg-red-50 dark:bg-red-900/30 rounded-lg'>Error: Please provide a valid Borrower ID.</div>"

    # DB expects integers, UI sometimes sends "B-0001"
    try:
        bid = int(raw_bid.replace("B-", "").replace("b-", ""))
    except ValueError:
        return f"<div class='text-red-500 text-sm font-bold p-4 bg-red-50 dark:bg-red-900/30 rounded-lg'>Invalid ID format: {raw_bid}</div>"

    try:
        from sentinel.credit.models import (
            EADModel,
            ExpectedLossEngine,
            LGDModel,
            XGBoostPDModel,
        )
        from sentinel.fraud.detector import FraudDetector
        from sentinel.fraud.features import (
            engineer_fraud_features,
            get_fraud_feature_columns,
        )

        borr = pd.read_sql(f"SELECT * FROM borrowers WHERE borrower_id = {bid}", engine.connect())
        if len(borr) == 0:
            return f"<div class='text-gray-500 dark:text-gray-400 font-bold p-6 text-center bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700'>No records found for identity: <code>{raw_bid}</code></div>"

        loans = pd.read_sql(f"SELECT * FROM loans WHERE borrower_id = {bid}", engine.connect())
        txns = pd.read_sql(f"SELECT * FROM transactions WHERE card_id = {bid}", engine.connect())

        all_loans = pd.read_sql("SELECT * FROM loans JOIN borrowers ON loans.borrower_id = borrowers.borrower_id", engine.connect())
        all_txns = pd.read_sql("SELECT * FROM transactions LIMIT 5000", engine.connect())
        if "borrower_id" in all_txns.columns:
            all_txns = all_txns.rename(columns={"borrower_id": "card_id"})

        features = ["fico_score", "annual_income", "emp_length_years", "loan_amount", "interest_rate", "dti", "is_secured"]

        # Credit pipeline
        el_dollars, pd_val = 0.0, 0.0
        if len(loans) > 0 and len(all_loans) > 20:
            pd_model = XGBoostPDModel(max_depth=3, n_estimators=20)
            pd_model.fit(all_loans[features], all_loans["is_default"])
            defaults = all_loans[all_loans["is_default"] == 1]
            if len(defaults) > 5:
                lgd_model = LGDModel(max_depth=2)
                lgd_model.fit(defaults[features], defaults["lgd"])
                ead_model = EADModel(max_depth=2)
                ead_model.fit(defaults[features], defaults["ead_factor"])
                loss_engine = ExpectedLossEngine(pd_model, lgd_model, ead_model)
                b_data = pd.merge(loans, borr, on="borrower_id")[features]
                el_df = loss_engine.predict_expected_loss(b_data, loans["loan_amount"])
                el_dollars = el_df["Expected_Loss"].sum()
                pd_val = el_df["PD"].mean()

        # Fraud pipeline
        fraud_exposure, fraud_prob, flagged_tx = 0.0, 0.0, 0
        if len(all_txns) > 20:
            featured_all = engineer_fraud_features(all_txns)
            f_model = FraudDetector(max_depth=3, n_estimators=20)
            f_model.fit(featured_all[get_fraud_feature_columns()], featured_all["is_fraud"])
            if len(txns) > 0:
                my_tx = txns.copy()
                if "borrower_id" in my_tx.columns:
                    my_tx = my_tx.rename(columns={"borrower_id": "card_id"})
                combined = pd.concat([all_txns, my_tx], ignore_index=True)
                my_featured = engineer_fraud_features(combined)
                my_featured = my_featured[my_featured["card_id"] == bid]
                if len(my_featured) > 0:
                    f_preds = f_model.predict_proba(my_featured[get_fraud_feature_columns()])
                    my_featured = my_featured.copy()
                    my_featured["fraud_prob"] = f_preds
                    high_risk = my_featured[my_featured["fraud_prob"] > 0.6]
                    fraud_exposure = high_risk["amount"].sum()
                    flagged_tx = len(high_risk)
                    fraud_prob = float(my_featured["fraud_prob"].mean())

        total_risk = el_dollars + fraud_exposure
        risk_level = "CRITICAL" if total_risk > 5000 else "ELEVATED" if total_risk > 1000 else "LOW"
        risk_badge_color = "bg-red-100 dark:bg-red-900/40 text-red-700 border-red-200 dark:border-red-800/50" if risk_level == "CRITICAL" else "bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 border-yellow-200 dark:border-yellow-800/50" if risk_level == "ELEVATED" else "bg-green-100 dark:bg-green-900/40 text-green-700 border-green-200 dark:border-green-800/50"
        b = borr.iloc[0]

        return f"""
        <div class="animate-in fade-in zoom-in duration-300">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
                    <div class="w-14 h-14 rounded-full bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center mx-auto mb-4"><i data-lucide="user" class="w-7 h-7 text-indigo-600"></i></div>
                    <h4 class="text-center font-bold text-gray-900 dark:text-white text-lg mb-1">{bid}</h4>
                    <div class="flex justify-center mb-5"><span class="px-2 py-0.5 text-[10px] font-bold rounded border {risk_badge_color}">{risk_level} RISK</span></div>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between py-1 border-b border-gray-100 dark:border-gray-800"><span class="text-gray-500 dark:text-gray-400">FICO</span><span class="font-bold">{b['fico_score']}</span></div>
                        <div class="flex justify-between py-1 border-b border-gray-100 dark:border-gray-800"><span class="text-gray-500 dark:text-gray-400">Income</span><span class="font-bold">${b['annual_income']:,.0f}</span></div>
                        <div class="flex justify-between py-1 border-b border-gray-100 dark:border-gray-800"><span class="text-gray-500 dark:text-gray-400">DTI</span><span class="font-bold">{loans['dti'].mean() if len(loans) > 0 else 0.0:.1f}%</span></div>
                        <div class="flex justify-between py-1 border-b border-gray-100 dark:border-gray-800"><span class="text-gray-500 dark:text-gray-400">Loans</span><span class="font-bold">{len(loans)}</span></div>
                        <div class="flex justify-between py-1"><span class="text-gray-500 dark:text-gray-400">TXNs</span><span class="font-bold">{len(txns)}</span></div>
                    </div>
                </div>
                <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 col-span-2">
                    <h4 class="font-bold text-gray-900 dark:text-white mb-5 flex items-center gap-2"><i data-lucide="activity" class="w-5 h-5 text-red-500"></i> Unified Risk Exposure</h4>
                    <div class="text-5xl font-black text-gray-900 dark:text-white mb-1">${total_risk:,.0f}</div>
                    <p class="text-xs text-gray-400 dark:text-gray-500 mb-6 uppercase tracking-wider">Total Value at Risk (Credit + Fraud)</p>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-red-50 dark:bg-red-900/30/50 p-4 rounded-xl border border-red-100 dark:border-red-900/50">
                            <div class="text-xs font-bold text-red-800 dark:text-red-300 uppercase tracking-wider mb-2 flex items-center gap-2"><i data-lucide="briefcase" class="w-4 h-4"></i> Credit Default Risk</div>
                            <div class="text-2xl font-bold text-red-600 mb-1">${el_dollars:,.0f}</div>
                            <div class="text-[10px] text-red-500">{pd_val:.1%} Avg PD | EL = PD × LGD × EAD</div>
                        </div>
                        <div class="bg-orange-50/50 dark:bg-orange-900/20 p-4 rounded-xl border border-orange-100 dark:border-orange-900/50">
                            <div class="text-xs font-bold text-orange-800 dark:text-orange-300 uppercase tracking-wider mb-2 flex items-center gap-2"><i data-lucide="shield-alert" class="w-4 h-4"></i> Fraud Transaction Risk</div>
                            <div class="text-2xl font-bold text-orange-600 mb-1">${fraud_exposure:,.0f}</div>
                            <div class="text-[10px] text-orange-500">{flagged_tx} flagged TXNs | Avg Fraud Score: {fraud_prob:.1%}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <script>lucide.createIcons();</script>
        """
    except Exception:
        import traceback
        return f"<div class='text-red-500 text-xs font-bold p-4 bg-red-50 dark:bg-red-900/30 rounded'><pre>{traceback.format_exc()}</pre></div>"


    content = '''
    <div class="mb-8">
        <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Portfolio VaR</h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Value at Risk (VaR) and Expected Shortfall (ES) Monte Carlo Backtesting.</p>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <h3 class="font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2"><i data-lucide="line-chart" class="w-5 h-5 text-indigo-500"></i> Simulation Parameters</h3>
            <p class="text-xs text-gray-500 dark:text-gray-400 mb-4">Configure the rolling window and VaR method for the Sentinel equity portfolio (SPY, AAPL, MSFT).</p>
            
            <form hx-post="/ui/api/engine/train_market" hx-target="#market-results" hx-indicator="#market-spinner" class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div class="col-span-2">
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Simulation Method</label>
                        <select name="method" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="parametric">Delta-Normal (Parametric)</option><option value="historical">Historical Simulation</option><option value="monte_carlo" selected>Monte Carlo (Geometric Brownian)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Confidence Level</label>
                        <select name="confidence_level" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="0.90">90.0% VaR</option><option value="0.95" selected>95.0% VaR</option><option value="0.99">99.0% VaR (Extreme Tail)</option></select>
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Rolling Window</label>
                        <select name="window_size" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="126">126 Days (6m)</option><option value="252" selected>252 Days (1yr)</option><option value="504">504 Days (2yr)</option></select>
                    </div>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-2">
                    <i data-lucide="zap" class="w-4 h-4"></i> Run Backtest
                </button>
            </form>
        </div>
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col justify-center items-center text-center">
            <div id="market-spinner" class="htmx-indicator hidden flex-col items-center">
                <i data-lucide="loader-2" class="w-8 h-8 text-indigo-600 animate-spin mb-3"></i>
                <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Executing Rolling Backtest...</p>
                <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">Calculating Kupiec POF Statistics</p>
            </div>
            <div id="market-results" class="w-full h-full min-h-[200px]">
                <div class="text-gray-400 dark:text-gray-500 font-medium text-sm flex flex-col items-center justify-center h-full">
                    <i data-lucide="bar-chart-2" class="w-8 h-8 mb-2 opacity-50"></i>
                    Awaiting Model Execution...
                </div>
            </div>
        </div>
    </div>
    '''
    if "hx-request" in request.headers: return content
    return get_base_html(content, "market", "Risk Engines > Portfolio VaR")

@app.get("/ui/view/importer", response_class=HTMLResponse)
def view_importer(request: Request):
    content = """
    <div class="mb-8">
        <div class="flex items-center gap-3 mb-2">
            <h2 class="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">Data Ingestion</h2>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">CSV &rarr; SQLite</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Schema Mapping</span>
                <span class="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded border border-indigo-100 dark:border-indigo-800/50">Append Mode</span>
            </div>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Bulk upload historical data arrays via CSV.</p>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="upload-cloud" class="w-5 h-5 text-indigo-500"></i> Import Configuration</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Upload CSV files to append data to specific tables.</p>
            </div>
            
            <form hx-post="/ui/api/db/upload" hx-target="#upload-status" hx-encoding="multipart/form-data" class="space-y-4">
                <div>
                    <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">Target Table</label>
                    <select name="target_table" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer"><option value="borrowers">Borrowers Table</option><option value="loans">Loans Table</option><option value="transactions">Transactions Table</option></select>
                </div>
                <div>
                    <label class="block text-[10px] font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-1">CSV File</label>
                    <input type="file" name="file" accept=".csv" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 shadow-sm transition-colors cursor-pointer" required>
                </div>
                <button type="submit" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded shadow-sm flex justify-center items-center gap-2 mt-4">
                    <i data-lucide="arrow-up-circle" class="w-4 h-4"></i> Execute Ingestion
                </button>
            </form>
            <div id="upload-status" class="mt-4"></div>
        </div>
        
        <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
            <div class="border-b border-gray-100 dark:border-gray-800 pb-3 mb-4">
                <h3 class="font-bold text-gray-900 dark:text-white flex items-center gap-2"><i data-lucide="layout-list" class="w-5 h-5 text-indigo-500"></i> Schema Reference</h3>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Expected columns for successful ingestion mapping.</p>
            </div>
            <div class="space-y-4 text-xs">
                <div>
                    <h4 class="font-bold text-gray-900 dark:text-white mb-1">Borrowers Table</h4>
                    <p class="text-gray-500 dark:text-gray-400 font-mono bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-100 dark:border-gray-800">borrower_id, fico_score, annual_income, emp_length_years</p>
                </div>
                <div>
                    <h4 class="font-bold text-gray-900 dark:text-white mb-1">Loans Table</h4>
                    <p class="text-gray-500 dark:text-gray-400 font-mono bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-100 dark:border-gray-800">loan_id, borrower_id, loan_amount, interest_rate, is_default, ead_factor, term_months</p>
                </div>
                <div>
                    <h4 class="font-bold text-gray-900 dark:text-white mb-1">Transactions Table</h4>
                    <p class="text-gray-500 dark:text-gray-400 font-mono bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-100 dark:border-gray-800">txn_id, borrower_id, amount, category, is_fraud, day, hour</p>
                </div>
            </div>
        </div>
    </div>
    <script>lucide.createIcons();</script>
    """
    if "hx-request" in request.headers: return content
    return get_base_html(content, "importer", "Data Infrastructure > Data Ingestion")

