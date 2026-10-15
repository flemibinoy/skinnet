"""Restrained, accessible presentation styles; no synthetic research graphics."""

from html import escape

CSS = """<style>
:root{--ink:#20212b;--muted:#606575;--accent:#49328a;--line:#e4e6ed;}
.stApp{background:#f7f8fb;color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
[data-testid="stHeader"]{display:none;}
[data-testid="stMainBlockContainer"]{max-width:1240px;padding:2rem 2.25rem 2.5rem;}
h1,h2,h3{color:var(--ink)!important;letter-spacing:-.025em;font-weight:600!important;}
h1{font-size:2rem!important;padding-top:0!important;line-height:1.2!important;}
h2{font-size:1.25rem!important;}h3{font-size:1.05rem!important;}
p,label{color:var(--ink);}[data-testid="stCaptionContainer"]{color:var(--muted);}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;gap:16px;}
.brand{font-size:23px;font-weight:700;letter-spacing:-.7px;color:var(--accent);}
.brand small{font-weight:400;color:var(--muted);font-size:12px;letter-spacing:0;margin-left:14px;}
.prototype{font-size:11px;color:#59477b;padding:5px 10px;background:#f1edf8;border-radius:6px;}
[data-testid="stRadioGroup"]{display:flex!important;flex-direction:row!important;flex-wrap:wrap;gap:4px;padding:5px;background:#eceef4;border:1px solid #e1e3eb;border-radius:10px;width:fit-content;margin:8px 0 18px;}
[data-testid="stRadioGroup"]>div{width:auto!important;}
[data-testid="stRadioOption"]{padding:8px 13px;border-radius:7px;}
[data-testid="stRadioOption"]>div>div:first-child{display:none!important;}
[data-testid="stRadioOption"][data-selected="true"]{background:white;box-shadow:0 1px 4px #20212b0e;}
[data-testid="stRadioOption"] p{font-size:13px!important;color:#464b5a;}
[data-testid="stRadioOption"][data-selected="true"] p{color:var(--accent);font-weight:600;}
[data-testid="stRadioOption"]:focus-within{outline:2px solid #7357af;outline-offset:2px;}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:12px;border-color:var(--line);}
[data-testid="stFileUploader"]{border:1px dashed #b9b3cb;border-radius:12px;background:white;padding:12px;}
[data-testid="stFileUploaderDropzone"]{background:white;}
[data-testid="stImage"] img{border-radius:8px;}
[data-testid="stMetric"]{padding:14px 16px;border:1px solid var(--line);border-radius:10px;background:#fff;}
[data-testid="stMetricValue"]{font-size:1.6rem!important;font-weight:600;color:var(--ink);}
[data-testid="stMetricLabel"]{color:var(--muted);}
.stButton>button{border-radius:7px;min-height:39px;font-size:13px;}
.stButton>button[kind="primary"]{background:var(--accent);border-color:var(--accent);color:#fff;}
.stButton>button:focus-visible{outline:2px solid #7357af;outline-offset:3px;}
[data-testid="stBaseButton-primary"]:not(:disabled) p{color:#fff!important;}
[data-testid="stBaseButton-primary"]:disabled{background:#edeef2!important;border-color:#e0e2e8!important;}
[data-testid="stBaseButton-primary"]:disabled p{color:#717784!important;}
[data-testid="stAlert"]{border-radius:9px;}
[data-testid="stExpander"]{background:#fff;border-radius:9px;border-color:var(--line);}
.empty{padding:26px 22px;border:1px solid var(--line);border-radius:12px;background:white;}
.empty strong{display:block;font-size:16px;margin-bottom:7px;color:var(--ink);}
.empty p{font-size:13px;color:var(--muted);line-height:1.6;margin:0;}
.notice{padding:13px 16px;border:1px solid #e2daef;border-left:4px solid #7155a4;background:#f4f1f9;font-size:13px;color:#403250;border-radius:0 9px 9px 0;margin:8px 0 18px;line-height:1.6;}
.footer{border-top:1px solid var(--line);margin-top:28px;padding-top:14px;font-size:11px;color:var(--muted);line-height:1.6;}
.prob-row{margin:10px 0;}.prob-label{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;color:#454957;}.prob-track{height:7px;border-radius:4px;background:#eef0f4;overflow:hidden;}.prob-fill{height:100%;border-radius:4px;background:#6f54a3;}
@media(max-width:820px){[data-testid="stMainBlockContainer"]{padding:1.5rem 1.15rem 2rem;}.brand small{display:none;}[data-testid="stRadioGroup"]{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr));width:100%;}[data-testid="stRadioOption"]{padding:8px;white-space:normal;}h1{font-size:1.65rem!important;}.prototype{max-width:145px;}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important;}}
</style>"""


def empty_state(title, description):
    return f'<div class="empty"><strong>{escape(title)}</strong><p>{escape(description)}</p></div>'


def probability_bars(probabilities):
    rows = []
    for label, p in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        percentage = min(100, max(0, float(p) * 100))
        rows.append(
            f'<div class="prob-row"><div class="prob-label"><span>{escape(label.upper())}</span><span>{percentage:.1f}%</span></div><div class="prob-track"><div class="prob-fill" style="width:{percentage:.4f}%"></div></div></div>'
        )
    return "".join(rows)
