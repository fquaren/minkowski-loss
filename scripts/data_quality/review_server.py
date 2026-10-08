#!/usr/bin/env python
"""Local review page for the v4 screen: the 100 radar-days of `quality_v4/review.csv` (exclude /
keep / unsure) and the 144 rule examples of `calib/rule_gallery/index.csv` (artefact / real
rain / mixed / unclear). Every decision is written straight into the CSV; a backup of each CSV
is taken once at start-up.

Stdlib only, bound to 127.0.0.1. From a laptop, tunnel to the node, e.g.

    ssh -N -L 8765:localhost:8765 -J <login node> node34.octopoda
    # then open http://localhost:8765

    python scripts/data_quality/review_server.py --port 8765

Claude's provisional gallery verdicts are kept in `index_provisional.csv` (created from
index.csv on the first start, while every row is still provisional), so the page can show them
next to yours. Images: `gallery/*.png` for radar-days, `calib/rule_gallery/crops/NNN.jpg` (one
crop of `rule_gallery.pdf` per example) for the gallery.
"""

import argparse
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

Q = "/home/fquareng/work/data/extremes/OPERA/quality_v4"
HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.join(Q, "review.csv")
INDEX = os.path.join(Q, "calib", "rule_gallery", "index.csv")
PROV = os.path.join(Q, "calib", "rule_gallery", "index_provisional.csv")
CROPS = os.path.join(Q, "calib", "rule_gallery", "crops")
LOCK = threading.Lock()


def num(v, nd=3):
    try:
        f = float(v)
        return None if f != f else round(f, nd)
    except (TypeError, ValueError):
        return None


def items():
    r = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    p = pd.read_csv(PROV, dtype=str, keep_default_na=False)
    R = [dict(id="r%03d" % int(x["rank"]), rank=int(x["rank"]), radar=x["radar"], loc=x["location"],
              country=x["country"], day=x["day"], score=num(x["score"], 2), reason=x["reason"],
              f150=num(x["f150"], 5), ceil=num(x["ceil_px"], 0), nflag=num(x["n_flag"], 0),
              iso31=num(x["iso31"], 0), n31=num(x["n31"], 0), maxr2=num(x["max_r2"], 2),
              ej=num(x["max_ejump"], 2), bj=num(x["max_bjump"], 2), event=x["event_day"],
              img="img/" + x["image"]) for _, x in r.iterrows()]
    G = [dict(id="g%03d" % (i + 1), rule=x["rule"], n=int(x["n"]), draw=x["draw"], day=x["day"],
              ts=x["timestamp"], row=int(x["row"]), col=int(x["col"]), cat=x["category"],
              px=int(float(x["px"])), pv=x["verdict"], pnote=x["note"], img="crop/%03d.jpg" % (i + 1))
         for i, x in p.iterrows()]
    return {"R": R, "G": G}


def state():
    r = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    g = pd.read_csv(INDEX, dtype=str, keep_default_na=False)
    R = {"r%03d" % int(x["rank"]): {"decision": x["decision"], "note": x["note"]}
         for _, x in r.iterrows() if x["decision"]}
    G = {"g%03d" % (i + 1): {"verdict": x["verdict"], "note": x["note"]}
         for i, x in g.iterrows() if x["reviewer"] == "researcher"}
    return {"R": R, "G": G}


def decide(tab, id_, body):
    with LOCK:
        if tab == "R":
            df = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
            i = df.index[df["rank"].astype(int) == int(id_[1:])][0]
            df.at[i, "decision"] = body.get("decision", "")
            df.at[i, "note"] = body.get("note", "")
            path = REVIEW
        else:
            df = pd.read_csv(INDEX, dtype=str, keep_default_na=False)
            i = int(id_[1:]) - 1
            if body.get("rule") and body["rule"] != df.at[i, "rule"]:
                raise ValueError("row mismatch")
            if body.get("verdict"):
                df.at[i, "verdict"] = body["verdict"]
                df.at[i, "reviewer"] = "researcher"
            if body.get("note"):
                df.at[i, "note"] = body["note"]
            path = INDEX
        tmp = path + ".part"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)


class H(BaseHTTPRequestHandler):
    page = b""

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send(200, H.page, "text/html; charset=utf-8")
        if p == "/api/state":
            return self._send(200, json.dumps(state()).encode(), "application/json")
        for prefix, root in (("/img/", Q), ("/crop/", CROPS)):
            if p.startswith(prefix):
                f = os.path.realpath(os.path.join(root, p[len(prefix):]))
                if f.startswith(os.path.realpath(root) + os.sep) and os.path.isfile(f):
                    ctype = "image/png" if f.endswith(".png") else "image/jpeg"
                    return self._send(200, open(f, "rb").read(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/decide":
            return self._send(404, b"not found", "text/plain")
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            decide(d["tab"], d["id"], d["body"])
            self._send(200, b"ok", "text/plain")
        except Exception as e:                                         # noqa: BLE001
            self._send(400, str(e).encode(), "text/plain")


def main():
    global REVIEW, INDEX, PROV, CROPS, Q
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--q", default=Q, help="quality_v4 directory (a copy, for testing)")
    a = ap.parse_args()
    Q = a.q
    REVIEW = os.path.join(Q, "review.csv")
    INDEX = os.path.join(Q, "calib", "rule_gallery", "index.csv")
    PROV = os.path.join(Q, "calib", "rule_gallery", "index_provisional.csv")
    CROPS = os.path.join(Q, "calib", "rule_gallery", "crops")
    if not os.path.exists(PROV):
        g = pd.read_csv(INDEX, dtype=str, keep_default_na=False)
        assert (g.reviewer == "claude-provisional").all(), "index.csv already edited; restore index_provisional.csv first"
        shutil.copy2(INDEX, PROV)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for f in (REVIEW, INDEX):
        shutil.copy2(f, f"{f}.bak_{stamp}")
    tpl = open(os.path.join(HERE, "review_page.html")).read()
    data = json.dumps(items()).replace("</", "<\\/")
    html = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"></head><body>'
            '<script>window.REVIEW_API="/api";</script>' + tpl.replace("__DATA__", data) + "</body></html>")
    H.page = html.encode()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"[review] http://localhost:{a.port}  (backups *.bak_{stamp}); Ctrl-C to stop", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
