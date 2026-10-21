# Why the screen misses it: rsjas_2023

Days 20230310, 20230722; mask 24,352 px; pixel-frames with raw >= 31 mm/h inside it. Replay of the v4 hourly chain (`HourlyDay`, no advection). Columns 'left: ...' count the pixel-frames still >= 31 after the chain that meet that rule's condition (pixel-hours for the burst columns).

| count | 20230310 | 20230722 |
|---|---|---|
| raw_ge_u | 216594 | 196972 |
| changed_clean_frame | 52046 | 46427 |
| footprint | 28615 | 6925 |
| ceiling | 23957 | 22989 |
| ring | 0 | 0 |
| ray | 0 | 0 |
| persistence | 224 | 9 |
| unsupported | 1217 | 6790 |
| burst | 21345 | 6156 |
| stationary | 0 | 0 |
| refused (too large to repair, left) | 20522 | 55722 |
| left_ge_u | 125722 | 124354 |
| left: spike-isolated | 819 | 275 |
| left: n_peak >= p1_min | 94 | 5 |
| left: burst share & rate | 49209 | 58845 |
| left: burst iso median | 496 | 237 |
| left: >= 10 with no echo nearby (prev/next) | 1933 | 3471 |
| left: raw > 500 | 21864 | 52251 |
| left: final > 500 | 12184 | 51484 |
| left: final > 500, coded refused | 12184 | 51484 |
| left: final 31-150 | 83221 | 38782 |
| left: on the ceiling mask | 6383 | 4304 |
| left: on the ceiling mask, coded refused | 202 | 11 |

## Structure of the noise

- Share of the mask >= 31 mm/h per frame: median 0.0002, 90th percentile 0.3905, frames with any: 67%.
- Rain around it (50 km ring outside the mask, share >= 1 mm/h): median 0.046; in frames where the mask has > 0.5% of pixels >= 31: 0.065.
- Connected components of >= 31 px touching the mask: 25,052; size median 2 px, 90th percentile 8, single pixels 48%.
- Commonest final values left >= the level (mm/h): 53.71 (479), 35.26 (411), 98.9 (327), 98.28 (298), 65.57 (288), 87.8 (270), 31575.9 (256), 67.22 (251)

