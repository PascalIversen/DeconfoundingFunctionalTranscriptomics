"""Reproducibly assign every gene set in the figure to a display class, using
ONLY official category sources (no hand-curated per-set list):

  * Human Gene Atlas sets            -> lineage          (tissue gene lists)
  * KEGG pathways  -> KEGG BRITE hierarchy (br08901)     official category
  * MSigDB Hallmark sets -> Liberzon 2015 process category (Table 1)

Display-class rule (category -> class), applied uniformly and documented:

  lineage   : Human Gene Atlas (tissue signatures)
  context   : KEGG {Human Diseases (incl. cancer-type pathways e.g. "Bladder
              cancer"), Organismal Systems, Cellular Processes:Cellular
              community / Cell motility, Environmental Information
              Processing:Signaling molecules and interaction}
              + Hallmark {immune, development, cellular component}
  intrinsic : KEGG {Metabolism, Genetic Information Processing,
              Cellular Processes:Cell growth&death / Transport&catabolism,
              Environmental Information Processing:Signal transduction}
              + Hallmark {proliferation, DNA damage, metabolic, pathway, signaling}

(The one deliberate choice is that intracellular *signal transduction* / Hallmark
"signaling" -> intrinsic; flip those two category rules to move it to context.)

Refetch the official inputs to reproduce:
  curl -s "https://rest.kegg.jp/get/br:br08901" -o gene_sets/kegg_brite.keg
  curl -s ".../h.all.v2024.1.Hs.symbols.gmt" -o gene_sets/hallmark_msigdb_official.gmt
  Hallmark process categories: Liberzon 2015 Table 1 -> gene_sets/hallmark_categories.tsv
"""
from pathlib import Path

GS = Path(__file__).resolve().parent / "gene_sets"


def gmt_sets(p):
    out = {}
    for ln in (GS / p).read_text().splitlines():
        t = ln.rstrip("\n").split("\t")
        if len(t) > 2:
            out[t[0]] = [g for g in t[2:] if g]
    return out


# ---- official KEGG BRITE: pathway name -> (A category, B subcategory) ----
def load_brite():
    cat = {}
    A = B = None
    for ln in (GS / "kegg_brite.keg").read_text().splitlines():
        if not ln or ln[0] not in "ABC":
            continue
        if ln[0] == "A":
            A = ln[1:].strip()
        elif ln[0] == "B":
            B = ln[1:].strip()
        elif ln[0] == "C":
            rest = ln[1:].strip().split(None, 1)
            if len(rest) == 2:
                cat[rest[1].strip().lower()] = (A, B)
    return cat


def kegg_class(A, B):
    if A in ("Metabolism", "Genetic Information Processing"):
        return "intrinsic"
    if A in ("Organismal Systems", "Human Diseases"):
        return "context"
    if A == "Environmental Information Processing":
        return "context" if "signaling molecules" in B.lower() else "intrinsic"
    if A == "Cellular Processes":
        b = B.lower()
        return "context" if ("community" in b or "motility" in b) else "intrinsic"
    return "context"


# KEGG renamed/removed some pathways between the Enrichr gmt version and current
# BRITE; reconcile by KEGG pathway ID, taking each one's official KEGG category.
KEGG_VERSION_ALIAS = {
    "rna transport": ("Genetic Information Processing", "Translation"),        # 03013
    "phagosome": ("Cellular Processes", "Transport and catabolism"),          # 04145
}


def fuzzy_brite(name, brite):
    """Match renamed pathways (e.g. 'Autophagy' -> 'Autophagy - animal')."""
    g = name.lower()
    gt = set(g.split())
    cands = [b for b in brite if b.startswith(g) or g.startswith(b) or gt.issubset(set(b.split()))]
    return brite[min(cands, key=len)] if cands else None


# ---- official Hallmark: process category (Liberzon 2015) ----
def norm(s):
    return s.upper().replace("HALLMARK_", "").replace("_", "").replace("DOWN", "DN")


def load_hallmark():
    cat = {}
    for ln in (GS / "hallmark_categories.tsv").read_text().splitlines():
        if ln.startswith("#") or "\t" not in ln:
            continue
        name, c = ln.split("\t")[:2]
        cat[norm(name)] = c.strip()
    official = {norm(k): set(v) for k, v in gmt_sets("hallmark_msigdb_official.gmt").items()}
    return cat, official


def hm_class(cat):
    return "context" if cat in ("immune", "development", "cellular component") else "intrinsic"


def main():
    kegg_g = gmt_sets("kegg.gmt")
    hall_g = gmt_sets("hallmark.gmt")
    hga_g = gmt_sets("human_gene_atlas.gmt")
    brite = load_brite()
    hm_cat, hm_official = load_hallmark()

    rows, unmatched_kegg, unmatched_hm = [], [], []

    for s in hga_g:
        rows.append((s, "HGA", "tissue signature", "lineage"))

    for s in kegg_g:
        hit = brite.get(s.lower()) or KEGG_VERSION_ALIAS.get(s.lower()) or fuzzy_brite(s, brite)
        if hit is None:
            unmatched_kegg.append(s)
            rows.append((s, "KEGG", "UNMATCHED", "context"))
        else:
            A, B = hit
            rows.append((s, "KEGG", f"{A} / {B}", kegg_class(A, B)))

    for s, genes in hall_g.items():
        g = set(genes)
        best, bestj = None, 0.0
        for sysname, og in hm_official.items():
            j = len(g & og) / max(1, len(g | og))
            if j > bestj:
                best, bestj = sysname, j
        cat = hm_cat.get(best)
        if cat is None or bestj < 0.3:
            unmatched_hm.append((s, best, round(bestj, 2)))
            rows.append((s, "Hallmark", f"UNMATCHED({best},{bestj:.2f})", "intrinsic"))
        else:
            rows.append((s, "Hallmark", f"Hallmark:{cat}", hm_class(cat)))

    out = GS / "set_classes.tsv"
    with out.open("w") as f:
        f.write("set\tsource\tofficial_category\tclass\n")
        for r in sorted(rows):
            f.write("\t".join(map(str, r)) + "\n")

    from collections import Counter
    cnt = Counter(r[3] for r in rows)
    print(f"wrote {out.name}: {len(rows)} sets  classes={dict(cnt)}")
    print(f"unmatched KEGG: {len(unmatched_kegg)} ; unmatched Hallmark: {len(unmatched_hm)}")
    if unmatched_kegg:
        print("  KEGG unmatched (defaulted to context):", unmatched_kegg[:20])
    if unmatched_hm:
        print("  Hallmark unmatched:", unmatched_hm)
    # sanity: Bladder cancer + a few flagged sets
    by = {r[0]: r for r in rows}
    for s in ["Bladder cancer", "Proteoglycans in cancer", "Focal adhesion",
              "KRAS Signaling Up", "Epithelial Mesenchymal Transition",
              "Estrogen Response Early", "E2F Targets", "Glycolysis"]:
        if s in by:
            print(f"  {s:38s} -> {by[s][3]:9s} [{by[s][1]}: {by[s][2]}]")


if __name__ == "__main__":
    main()
