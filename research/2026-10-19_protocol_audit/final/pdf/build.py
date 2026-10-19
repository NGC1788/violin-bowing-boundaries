import re, html, subprocess, sys
CSS="""<style>@page{size:A4;margin:16mm 15mm}body{font-family:'Apple SD Gothic Neo',sans-serif;font-size:10.5pt;line-height:1.55;color:#222}
h1{font-size:19pt;margin:0 0 4pt}h2{font-size:13.5pt;border-bottom:2px solid #1f4e9c;color:#1f4e9c;margin:14pt 0 5pt}h3{font-size:11.5pt;margin:10pt 0 3pt}
table{border-collapse:collapse;margin:5pt 0;font-size:9.5pt}td,th{border:1px solid #bbb;padding:3pt 6pt}th{background:#eef2fa}
.box{background:#f3f6fb;border-left:4px solid #1f4e9c;padding:6pt 9pt;margin:6pt 0}img{width:100%;margin:4pt 0}.sub{color:#666}</style>"""
def inline(t):
    t=html.escape(t); t=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",t); t=re.sub(r"`(.+?)`",r"<code>\1</code>",t); return t
def md2html(md):
    out=[];inl=False;tab=[]
    def flush():
        nonlocal tab
        if tab:
            rows=[r for r in tab if not re.match(r"^\|[-| :]+\|$",r)]
            h="<table>"+"".join("<tr>"+"".join(f"<{'th' if i==0 else 'td'}>{inline(c.strip())}</{'th' if i==0 else 'td'}>" for c in r.strip('|').split('|'))+"</tr>" for i,r in enumerate(rows))+"</table>"
            out.append(h); tab=[]
    for line in md.splitlines():
        if line.startswith("|"): tab.append(line); continue
        flush()
        m=re.match(r"^(#{1,3}) (.*)",line)
        if m: out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>"); continue
        if re.match(r"^\s*([-*]|\d+\.) ",line):
            ind=1+(len(line)-len(line.lstrip()))//2; body=inline(re.sub(r"^\s*([-*]|\d+\.) ","",line))
            out.append("<li style='margin-left:%dem'>%s</li>"%(ind,body)); continue
        if line.startswith("> "): out.append(f"<div class=box>{inline(line[2:])}</div>"); continue
        if line.startswith("!["):
            src=re.search(r"\((.*)\)",line).group(1); out.append("<img src='%s'>"%src); continue
        if line.strip(): out.append(line if line.startswith("<span") else "<p>%s</p>"%inline(line))
    flush(); return "\n".join(out)
src,dst=sys.argv[1],sys.argv[2]
open(dst+".html","w").write("<html><head><meta charset=utf-8>"+CSS+"</head><body>"+md2html(open(src).read())+"</body></html>")
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome","--headless","--disable-gpu","--no-pdf-header-footer",f"--print-to-pdf={dst}","file://"+__import__('os').path.abspath(dst+".html")],capture_output=True)
