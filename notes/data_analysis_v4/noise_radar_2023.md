# The radar behind the 2023 disk-shaped noise in the raw >= 31 mm/h climatology

Anomaly: 24,352 px (97,408 km²), frequency > 1e-3 and > 5x the median of [2021, 2022, 2024, 2025]. Circle fitted to its western arc: centre row 540.0, col 1437.8, radius 248 km. Its straight eastern edge (col = -0.154 row + 1521) has its midpoint at row 541, col 1438: the edge is a diameter through the same point.

| odim | location | country | lat | lon | maxrange_km | km_to_arc_centre | km_to_edge_midpoint |
|---|---|---|---|---|---|---|---|
| rsjas | Jastrebac | Serbia | 43.39 | 21.44 | 400 | 4.058 | 2 |
| rskru | Krusevac | Serbia | 43.62 | 21.25 | 250 | 31.46 | 29.97 |
| rsnis | Nis | Serbia | 43.4 | 21.95 | 250 | 42.11 | 41.27 |
| rskuk | Kukavica | Serbia | 42.79 | 21.95 | 250 | 75.27 | 76.61 |
| rsbes | Besnjaja | Serbia | 44 | 21.05 | 250 | 75.32 | 73.76 |
| rscrn | Crni vrh | Serbia | 44.13 | 21.96 | 250 | 95.02 | 92.99 |

**rsjas (Jastrebac, Serbia)** is 4 km from the fitted centre; the next site is 31 km away. The radius matches the composite's 250 km range.

- In the composites' node lists (one sampled timestamp per month): {'2017': 3, '2018': 9, '2019': 12, '2020': 11, '2021': 8, '2022': 10, '2023': 6} months per year; in 2023: ['2023-01', '2023-02', '2023-06', '2023-07', '2023-08', '2023-11'].
- Radar-days excluded by rule 5: {'2019': 1, '2020': 7, '2021': 14, '2022': 34, '2023': 12}.
- On the suspect-radar page (`suspect_radars.pdf`, ownership of 2024) it carries no label because it is not in the 2024 node lists, so the 2024 geometry has no such radar.
