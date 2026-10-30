"""Visualize the CTI knowledge graph.

Reads data/cti_knowledge_graph.json and data/entities.jsonl and writes to figures/:
  entity_type_counts.png        mentions per entity type
  relation_counts.png           extracted relations per type and confidence
  graph_overview.png            all entities connected by extracted relations
  neighborhood_<name>.png       relations around the best-connected threat actors / malware
  graph_interactive.html        zoomable, draggable version (loads vis-network from a CDN)

Run from the cti_graph_rag directory:  python src/visualize_graph.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "figures"
RELATIONS = ["USES", "USES_TECHNIQUE", "EXPLOITS", "TARGETS"]

TYPE_COLORS = {
    "GROUP": "#d55e00", "MALWARE": "#cc3311", "TOOL": "#e69f00", "TECHNIQUE": "#009e73",
    "TACTIC": "#56b4e9", "CVE": "#7b3294", "ORG": "#0072b2", "SECTOR": "#6a8caf",
    "LOC": "#8c8c8c", "CON": "#bbbbbb", "DATE": "#dddddd",
}
REL_COLORS = {"USES": "#cc3311", "USES_TECHNIQUE": "#009e73", "EXPLOITS": "#7b3294", "TARGETS": "#0072b2"}
CONF_ALPHA = {"high": 1.0, "medium": 0.6, "low": 0.35}


def entity_type_of(node):
    return node.get("entity_type") or node.get("ner_type") or "CON"


def type_of_id(nid, node):
    t = entity_type_of(node)
    if nid.startswith("NORM_"):
        t = nid.split("_", 2)[1]
    return t


def load_relation_graph():
    g = json.load(open(DATA / "cti_knowledge_graph.json", encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    G = nx.MultiDiGraph()
    for e in g["edges"]:
        if e["relation"] not in RELATIONS:
            continue
        p = e["properties"]
        for nid in (e["source"], e["target"]):
            n = nodes[nid]
            G.add_node(nid, label=n.get("name") or n.get("mention"), etype=type_of_id(nid, n))
        G.add_edge(e["source"], e["target"], relation=e["relation"], confidence=p["confidence"],
                   document=p["document"], evidence=p["evidence_text"], date=p["publication_date"])
    return G


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def plot_entity_counts():
    counts = Counter(json.loads(l)["entity_type"] for l in open(DATA / "entities.jsonl", encoding="utf-8"))
    items = counts.most_common()[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=[TYPE_COLORS.get(k, "#999") for k, _ in items])
    for i, (_, v) in enumerate(items):
        ax.text(v + max(counts.values()) * 0.01, i, f"{v:,}", va="center", fontsize=9)
    ax.set_title("Entity mentions by type (120 reports)", loc="left", fontsize=12)
    ax.set_xlabel("mentions")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "entity_type_counts.png", dpi=160)
    plt.close(fig)


def plot_relation_counts(G):
    c = Counter((d["relation"], d["confidence"]) for _, _, d in G.edges(data=True))
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0] * len(RELATIONS)
    for conf, shade in (("high", "#1f4e79"), ("medium", "#6fa8d6"), ("low", "#c5dcef")):
        vals = [c[(r, conf)] for r in RELATIONS]
        ax.bar(RELATIONS, vals, bottom=bottom, label=conf, color=shade, width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, total in enumerate(bottom):
        ax.text(i, total + 1, str(total), ha="center", fontsize=9)
    ax.set_title("Extracted relations by type and confidence", loc="left", fontsize=12)
    ax.set_ylabel("relation instances")
    ax.legend(title="confidence", frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "relation_counts.png", dpi=160)
    plt.close(fig)


def packed_layout(G, per_row_width=26.0):
    """Lay out each connected component on its own and shelf-pack them left to right."""
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    pos, x, y, row_h = {}, 0.0, 0.0, 0.0
    for comp in comps:
        H = G.subgraph(comp).to_undirected()
        size = 1.6 if len(comp) <= 2 else 1.5 + (1.9 if len(comp) > 20 else 1.0) * len(comp) ** 0.5
        local = nx.spring_layout(nx.Graph(H), seed=7, k=1.8 / max(len(comp), 1) ** 0.5, iterations=300)             if len(comp) > 2 else {n: (i * 0.8 - 0.4, 0) for i, n in enumerate(comp)}
        if x + size > per_row_width:
            x, y, row_h = 0.0, y - row_h - 0.4, 0.0
        for n, (px, py) in local.items():
            pos[n] = (x + size / 2 + px * size / 2, y - size / 2 + py * size / 2)
        x += size + 0.4
        row_h = max(row_h, size)
    return pos


def draw(G, path, title, label_top=25, figsize=(14, 10), packed=False):
    pos = packed_layout(G) if packed else nx.spring_layout(
        nx.Graph(G.to_undirected()), seed=7, k=1.2 / max(len(G), 1) ** 0.5, iterations=200)
    deg = dict(G.degree())
    fig, ax = plt.subplots(figsize=figsize)
    for u, v, d in G.edges(data=True):
        ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(
            arrowstyle="-|>", color=REL_COLORS[d["relation"]], alpha=0.35 + 0.65 * CONF_ALPHA[d["confidence"]],
            lw=1.3, shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=0.08"))
    ax.scatter([pos[n][0] for n in G], [pos[n][1] for n in G], s=[35 + 18 * deg[n] for n in G],
               c=[TYPE_COLORS.get(G.nodes[n]["etype"], "#999") for n in G],
               edgecolors="white", linewidths=0.8, zorder=3)
    for n in sorted(G, key=lambda n: -deg[n])[:label_top]:
        ax.text(pos[n][0], pos[n][1] + 0.05, G.nodes[n]["label"][:28], ha="center", fontsize=7 if packed else 8,
                zorder=4, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    present_types = sorted({G.nodes[n]["etype"] for n in G})
    handles = [plt.Line2D([], [], marker="o", ls="", color=TYPE_COLORS.get(t, "#999"), label=t) for t in present_types]
    handles += [plt.Line2D([], [], color=c, lw=2, label=r) for r, c in REL_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.set_title(title, loc="left", fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def hub_nodes(G, n=3):
    cand = [x for x in G if G.nodes[x]["etype"] in ("GROUP", "MALWARE")]
    return sorted(cand, key=lambda x: -G.degree(x))[:n]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def write_interactive(G):
    nodes = [{"id": n, "label": d["label"], "value": 3 + G.degree(n), "group": d["etype"],
              "color": TYPE_COLORS.get(d["etype"], "#999")} for n, d in G.nodes(data=True)]
    edges = [{"from": u, "to": v, "arrows": "to", "color": {"color": REL_COLORS[d["relation"]], "opacity": CONF_ALPHA[d["confidence"]]},
              "title": f"{d['relation']} ({d['confidence']}) · {d['date']}<br>{d['evidence'][:220]}"}
             for u, v, d in G.edges(data=True)]
    legend = "".join(f'<span><i style="background:{c}"></i>{t}</span>' for t, c in TYPE_COLORS.items() if t not in ("CON", "DATE"))
    rel_legend = "".join(f'<span><b style="color:{c}">&#8212;</b> {r}</span>' for r, c in REL_COLORS.items())
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CTI Knowledge Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
<style>body{{margin:0;font:14px system-ui,sans-serif;color:#222}}#bar{{padding:10px 16px;border-bottom:1px solid #ddd}}
#bar span{{margin-right:14px;white-space:nowrap}}#bar i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
#g{{height:calc(100vh - 78px)}}</style></head><body>
<div id="bar"><strong>CTI knowledge graph</strong> &middot; hover an edge for its evidence sentence<br>{legend}<br>{rel_legend}</div>
<div id="g"></div><script>
const nodes=new vis.DataSet({json.dumps(nodes)}),edges=new vis.DataSet({json.dumps(edges)});
new vis.Network(document.getElementById('g'),{{nodes,edges}},{{nodes:{{shape:'dot',font:{{size:13}}}},
physics:{{solver:'forceAtlas2Based',stabilization:{{iterations:250}}}},interaction:{{hover:true,tooltipDelay:100}}}});
</script></body></html>"""
    (OUT / "graph_interactive.html").write_text(html, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    G = load_relation_graph()
    print(f"Relation graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    plot_entity_counts()
    plot_relation_counts(G)
    draw(G, OUT / "graph_overview.png", "CTI knowledge graph: each cluster is a set of entities linked by extracted relations",
         label_top=140, figsize=(22, 14), packed=True)
    for hub in hub_nodes(G):
        sub = G.subgraph({hub} | set(G.predecessors(hub)) | set(G.successors(hub))).copy()
        draw(sub, OUT / f"neighborhood_{slug(G.nodes[hub]['label'])}.png",
             f"Relations around {G.nodes[hub]['label']} ({sub.number_of_edges()} edges)",
             label_top=40, figsize=(11, 8))
    write_interactive(G)
    print("Wrote figures to", OUT)


if __name__ == "__main__":
    main()
