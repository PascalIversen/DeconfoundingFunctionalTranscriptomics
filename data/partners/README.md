# Functional-partner ground truth (Fig 4a)

- `CORUM.gmt`: CORUM protein-complex co-membership (1,658 complexes; Tsitsiridis
  et al. 2022). Bundled.
- **STRING v12**, one-hop neighbours with combined score ≥ 700. Not bundled
  (the raw `links.v12.0.txt.gz` is ~81 MB). Download from
  <https://string-db.org/> (v12.0, `9606.protein.links.v12.0.txt.gz` +
  `9606.protein.info.v12.0.txt.gz`) and map to HUGO symbols.

The derived per-item recovery AUROCs against these sets are already in
`../../fig04a_selfrank_partner/data/panel_newpanel_depmap*.csv`, so Fig 4a
reproduces without re-downloading STRING.
