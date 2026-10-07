"""Build a self-contained blind A/B listening test (one HTML file, audio embedded).

Each trial plays the same sentence from two models in random order. The listener
picks which sounds more natural AND which conveys the intended tone better, plus
an optional emotion tag. Results are shown at the end as JSON to paste into
results/<run_id>/ab_<listener>.json, then:

  python make_ab_test.py results/<run_id> --score

reports preference rate per model with a binomial 95% CI and p-value.

  python make_ab_test.py results/<run_id> --models eleven_v4 eleven_v4_turbo
"""
import argparse
import base64
import json
import random
from pathlib import Path

INTENT = {"question": "asking a genuine question", "question_open": "asking a genuine question",
          "apology": "sincere apology", "empathy": "warm and empathetic", "excited": "happy / excited",
          "info": "clear and neutral", "ack": "quick, friendly acknowledgement"}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind TTS A/B</title><style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;line-height:1.5}
.card{border:1px solid #ccc;border-radius:10px;padding:1rem;margin:1rem 0}
audio{width:100%;margin:.3rem 0} button{padding:.5rem 1rem;margin:.25rem;border-radius:6px;border:1px solid #888;cursor:pointer}
button.sel{background:#222;color:#fff} textarea{width:100%;height:12rem}</style></head><body>
<h2>Blind voice comparison</h2><p>Same sentence, two voices, random order. Use headphones.</p>
<div id="app"></div><script>
const T=__TRIALS__;const ans=[];let i=0;
function q(label,key,opts,t){return `<p><b>${label}</b><br>`+opts.map(o=>`<button onclick="pick(this,'${key}','${o}')">${o}</button>`).join('')+`</p>`}
function pick(b,k,v){ans[i][k]=v;[...b.parentNode.querySelectorAll('button')].forEach(x=>x.classList.remove('sel'));b.classList.add('sel')}
function render(){const el=document.getElementById('app');
 if(i>=T.length){el.innerHTML='<h3>Done, thank you!</h3><p>Copy this and send it back:</p><textarea>'+JSON.stringify(ans)+'</textarea>';return}
 const t=T[i];ans[i]={trial:t.id};
 el.innerHTML=`<div class="card"><p>${i+1} / ${T.length}</p><p><i>"${t.text}"</i></p><p>Intended tone: <b>${t.intent}</b></p>
 A<audio controls src="${t.a}"></audio>B<audio controls src="${t.b}"></audio>
 ${q('More natural?','natural',['A','B','Same'])}${q('Closer to the intended tone?','tone',['A','B','Same'])}
 ${q('Emotion of A','emo_a',['neutral','warm','happy','sad','flat/robotic'])}${q('Emotion of B','emo_b',['neutral','warm','happy','sad','flat/robotic'])}
 <button onclick="if(ans[i].natural&&ans[i].tone){i++;render()}else alert('Answer the first two')">Next</button></div>`}
render();</script></body></html>"""


def b64(path):
    return "data:audio/wav;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def build(run, models, per_item):
    rows = [json.loads(l) for l in (run / "latency.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok") and r["model"] in models and not r["category"].startswith("length")]
    corpus = {json.loads(l)["id"]: json.loads(l) for l in Path("corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    by = {}
    for r in rows:
        by.setdefault((r["item_id"], r["model"]), []).append(r)
    trials, key = [], {}
    for item in {r["item_id"] for r in rows}:
        if not all((item, m) in by for m in models):
            continue
        for k in range(per_item):
            ra, rb = (random.choice(by[(item, m)]) for m in models)
            if random.random() < .5:
                ra, rb = rb, ra
            tid = f"{item}#{k}"
            key[tid] = {"A": ra["model"], "B": rb["model"], "category": ra["category"]}
            trials.append({"id": tid, "text": corpus[item]["text"],
                           "intent": INTENT.get(ra["category"], "natural, as a support agent would say it"),
                           "a": b64(ra["wav"]), "b": b64(rb["wav"])})
    random.shuffle(trials)
    (run / "ab_test.html").write_text(PAGE.replace("__TRIALS__", json.dumps(trials)), encoding="utf-8")
    (run / "ab_key.json").write_text(json.dumps(key, indent=2))  # keep out of the listener's hands
    print(f"{len(trials)} trials -> {run / 'ab_test.html'}")


def score(run):
    from scipy.stats import binomtest
    key = json.loads((run / "ab_key.json").read_text())
    wins = {}
    for f in run.glob("ab_*.json"):
        if f.name == "ab_key.json":
            continue
        for a in json.loads(f.read_text()):
            k = key[a["trial"]]
            for q in ("natural", "tone"):
                if a.get(q) in ("A", "B"):
                    winner = k[a[q]]
                    loser = k["B" if a[q] == "A" else "A"]
                    for cat in (k["category"], "ALL"):
                        d = wins.setdefault((q, cat), {})
                        d[winner] = d.get(winner, 0) + 1
                        d.setdefault(loser, 0)
    for (q, cat), d in sorted(wins.items(), key=lambda x: (x[0][0], x[0][1] != "ALL", x[0][1])):
        (m1, w1), (m2, w2) = sorted(d.items())
        n = w1 + w2
        t = binomtest(w1, n)
        ci = t.proportion_ci()
        print(f"{q:<8} {cat:<14} {m1} preferred {w1}/{n} = {w1/n:.0%} "
              f"[{ci.low:.0%}-{ci.high:.0%}] p={t.pvalue:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--models", nargs=2, default=["eleven_v4", "eleven_v4_turbo"])
    ap.add_argument("--per-item", type=int, default=2)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    score(Path(a.run_dir)) if a.score else build(Path(a.run_dir), a.models, a.per_item)
