import random, csv, json, re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

random.seed(42)
np.random.seed(42)

PARTS = {
 "brake pad":     ["brake pad", "brake pads", "disc pad"],
 "air filter":    ["air filter", "hawa filter"],
 "oil filter":    ["oil filter", "engine oil filter"],
 "clutch plate":  ["clutch plate", "clutch kit"],
 "spark plug":    ["spark plug", "plug"],
 "headlight":     ["headlight", "head lamp", "headlamp"],
 "shock absorber":["shock absorber", "shocker", "suspension"],
 "wiper blade":   ["wiper blade", "wiper"],
 "battery":       ["battery", "car battery"],
 "radiator":      ["radiator", "coolant radiator"],
 "timing belt":   ["timing belt", "cam belt"],
 "side mirror":   ["side mirror", "sheesha"],
}
VEHICLES = ["swift", "alto", "baleno", "i20", "creta", "city", "innova", "bolero", "splendor", "pulsar"]
PART_NOISE = {
 "brake pad": ["break pad", "brek pad", "brake paad"],
 "air filter": ["air filtar", "ayr filter", "air filtr"],
 "oil filter": ["oil filtar", "oyl filter"],
 "clutch plate": ["clutch plait", "klatch plate", "clutch plat"],
 "spark plug": ["spark plag", "spaark plug"],
 "headlight": ["headlite", "head light", "hedlight"],
 "shock absorber": ["shock absorbar", "shok absorber", "shocker"],
 "wiper blade": ["wiper blad", "waiper blade"],
 "battery": ["battary", "batri", "battri"],
 "radiator": ["radiater", "radietar"],
 "timing belt": ["timming belt", "taiming belt"],
 "side mirror": ["side miror", "saide mirror", "sheesha"],
}
VEH_NOISE = {"swift":["sweft","swivt"],"alto":["alto","alto k10"],"baleno":["balleno","balino"],
 "i20":["i 20","i-20"],"creta":["kreta","cretta"],"city":["siti","city"],"innova":["inova","innowa"],
 "bolero":["bolaro","bolero"],"splendor":["splender","splendar"],"pulsar":["pulser","palsar"]}

BIKE_OK = {"brake pad","air filter","oil filter","clutch plate","spark plug","headlight","shock absorber","battery"}
catalog = []
pid = 0
for p in PARTS:
    for v in VEHICLES:
        if v in ("splendor","pulsar") and p not in BIKE_OK: continue
        if random.random() < 0.12: continue
        pid += 1
        catalog.append({"id": pid, "part": p, "vehicle": v,
                        "title": f"{p} for {v}",
                        "doc": f"{random.choice(PARTS[p])} {v} {random.choice(['genuine','aftermarket','OEM'])} {random.choice(['Bosch','Minda','Denso','Gabriel','Exide'])}"})
have = {(c["part"], c["vehicle"]): c["id"] for c in catalog}

TEMPLATES = [
 "bhai {p} {v} me lagega?",
 "{v} ke liye {p} milega kya?",
 "{p} {v} ka kitne ka hai",
 "mere {v} ki {p} chahiye, stock me hai?",
 "bhaiya {v} {p} available hai kya",
 "{v} me kaunsa {p} fit hoga yaar",
 "{p} for {v} price kya hai",
 "ek {p} chahiye {v} ke liye jaldi batao",
]
def make_query(part, veh, level):
    t = random.choice(TEMPLATES)
    if level == "clean":
        p, v = part, veh
    elif level == "light":
        p, v = random.choice(PARTS[part]), veh
    else:
        p, v = random.choice(PART_NOISE[part]), random.choice(VEH_NOISE[veh])
    return t.format(p=p, v=v)

queries = []
for level in ["clean","light","heavy"]:
    n = 0
    while n < 60:
        part, veh = random.choice(list(PARTS)), random.choice(VEHICLES)
        if veh in ("splendor","pulsar") and part not in BIKE_OK: continue
        gold = have.get((part, veh))
        queries.append({"q": make_query(part, veh, level), "level": level,
                        "gold_id": gold, "part": part, "vehicle": veh}); n += 1

with open("data/catalog.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id","part","vehicle","title","doc"]); w.writeheader(); w.writerows(catalog)
with open("data/queries.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["q","level","gold_id","part","vehicle"]); w.writeheader(); w.writerows(queries)

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
docs = [c["title"] + " " + c["doc"] for c in catalog]
bm25 = BM25Okapi([tok(d) for d in docs])
tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True)
D = tf.fit_transform(docs)

def rank_bm25(q):
    s = bm25.get_scores(tok(q)); return np.argsort(-s), s
def rank_char(q):
    s = (tf.transform([q]) @ D.T).toarray()[0]; return np.argsort(-s), s
def rank_hybrid(q):
    r1,_ = rank_bm25(q); r2,_ = rank_char(q)
    sc = np.zeros(len(docs))
    for r in (r1,r2):
        for pos, i in enumerate(r): sc[i] += 1/(60+pos)
    return np.argsort(-sc), sc

SYS = {"BM25 (word)": rank_bm25, "TF-IDF (char n-gram)": rank_char, "Hybrid (RRF)": rank_hybrid}

def boot_ci(x, n=2000):
    x = np.array(x, float)
    m = [np.random.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m,2.5), np.percentile(m,97.5)

ans = [q for q in queries if q["gold_id"] is not None]
un  = [q for q in queries if q["gold_id"] is None]
print(f"catalog={len(catalog)} products | answerable queries={len(ans)} | unanswerable={len(un)}\n")

results = {}
for name, fn in SYS.items():
    results[name] = {}
    for level in ["clean","light","heavy","ALL"]:
        qs = [q for q in ans if level=="ALL" or q["level"]==level]
        t1, t3 = [], []
        for q in qs:
            order = fn(q["q"])[0]
            ids = [catalog[i]["id"] for i in order[:3]]
            t1.append(int(ids[0]==q["gold_id"])); t3.append(int(q["gold_id"] in ids))
        lo, hi = boot_ci(t1)
        results[name][level] = {"n":len(qs),"top1":float(np.mean(t1)),"top1_ci":[float(lo),float(hi)],"top3":float(np.mean(t3))}
    hits = []
    for q in ans:
        o, s = fn(q["q"])
        if catalog[o[0]]["id"]==q["gold_id"]: hits.append(s[o[0]])
    thr = float(np.percentile(hits, 10))
    def top_score(q):
        o, s = fn(q["q"]); return s[o[0]]
    results[name]["abstain"] = {"threshold":thr,
        "unanswerable_wrongly_answered": float(np.mean([top_score(q)>=thr for q in un])),
        "answerable_kept": float(np.mean([top_score(q)>=thr for q in ans]))}

for name, r in results.items():
    print(name)
    for level in ["clean","light","heavy","ALL"]:
        x = r[level]; print(f"  {level:5s} n={x['n']:3d}  top1={x['top1']:.2f} [{x['top1_ci'][0]:.2f},{x['top1_ci'][1]:.2f}]  top3={x['top3']:.2f}")
    a = r["abstain"]; print(f"  abstain: unanswerable wrongly answered={a['unanswerable_wrongly_answered']:.2f}, answerable kept={a['answerable_kept']:.2f}")
json.dump(results, open("results/results.json","w"), indent=1)

print("\nSample failures (BM25 word, heavy noise):")
k=0
for q in ans:
    if q["level"]=="heavy":
        top = catalog[rank_bm25(q["q"])[0][0]]
        if top["id"]!=q["gold_id"]:
            print(f"  '{q['q']}' -> got '{top['title']}' (wanted {q['part']} for {q['vehicle']})"); k+=1
            if k==5: break
print("\nSample failures (Hybrid, heavy noise):")
k=0
for q in ans:
    if q["level"]=="heavy":
        top = catalog[rank_hybrid(q["q"])[0][0]]
        if top["id"]!=q["gold_id"]:
            print(f"  '{q['q']}' -> got '{top['title']}' (wanted {q['part']} for {q['vehicle']})"); k+=1
            if k==5: break
