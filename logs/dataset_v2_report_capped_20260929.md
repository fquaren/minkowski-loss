# v2 dataset report

budget 3,000,000; shares {'dry': 0.08, 'light': 0.12, 'moderate': 0.25, 'heavy': 0.25, 'tail': 0.3}; val/test 0.1/0.1; buffer 1 d; event buffer 1 d; seed 20260928

## Rejection by raw tile max (mm/h)

| raw max | tiles | rejected | unphysical | ray |
|---|---|---|---|---|
| [0.0, 0.1) | 6,551,981 | 0.00% | 0.00% | 0.00% |
| [0.1, 1.0) | 5,373,026 | 0.00% | 0.00% | 0.00% |
| [1.0, 10.0) | 11,572,872 | 0.18% | 0.00% | 0.18% |
| [10.0, 31.0) | 4,056,582 | 0.50% | 0.00% | 0.50% |
| [31.0, 89.0) | 1,785,530 | 0.45% | 0.00% | 0.45% |
| [89.0, 150.0) | 387,473 | 0.32% | 0.00% | 0.32% |
| [150.0, 500.0) | 214,412 | 0.61% | 0.00% | 0.61% |
| [500.0, inf) | 76,802 | 45.04% | 44.91% | 0.39% |

## Days

train: 3248 | test: 488 | val: 420 | drop: 189

min gap train <-> val: 2 day(s)
min gap train <-> test: 2 day(s)
min gap val <-> test: 2 day(s)

full_train: 2,351,676 patches (pool 22,446,312); dry 188,134, light 282,201, moderate 587,919, heavy 587,919, tail 705,503
full_val: 305,235 patches (pool 2,913,415); dry 24,419, light 36,628, moderate 76,309, heavy 76,309, tail 91,570
full_test: 343,088 patches (pool 3,274,713); dry 27,447, light 41,171, moderate 85,772, heavy 85,772, tail 102,926

light_train: 587,920 | extremes_train: 705,503
light_val: 76,308 | extremes_val: 91,570
light_test: 85,773 | extremes_test: 102,926
events_test: 33,468

## Events in test

| event | tiles | tiles >= 31 | max (mm/h) |
|---|---|---|---|
| hymex_iop6_20120924 | 1,370 | 150 | 273.4 |
| hymex_iop16_20121026 | 0 | 0 | nan |
| pentecost_ela_20140609 | 2,015 | 693 | 364.6 |
| muenster_20140728 | 576 | 315 | 151.2 |
| montpellier_20140929 | 288 | 46 | 205.0 |
| cannes_20151003 | 191 | 66 | 177.6 |
| elvira_friederike_201605 | 13,809 | 2,889 | 452.5 |
| poland_derecho_20170811 | 946 | 134 | 286.1 |
| aude_20181015 | 384 | 79 | 127.9 |
| alex_20201002 | 159 | 69 | 375.0 |
| swiss_hail_20210620 | 192 | 102 | 166.6 |
| moravia_tornado_20210624 | 96 | 62 | 318.9 |
| swiss_hail_20210628 | 190 | 91 | 181.4 |
| swiss_italy_hail_20210708 | 284 | 196 | 177.7 |
| bernd_20210713 | 1,534 | 356 | 451.6 |
| med_derecho_20220818 | 2,429 | 910 | 487.6 |
| emilia_romagna_20230502 | 0 | 0 | nan |
| emilia_romagna_20230516 | 0 | 0 | nan |
| italy_hail_20230719 | 153 | 143 | 486.7 |
| chauxdefonds_italy_20230724 | 658 | 476 | 492.8 |
| slovenia_20230804 | 1,130 | 573 | 150.0 |
| s_germany_20240601 | 1,536 | 562 | 146.2 |
| misox_20240621 | 192 | 127 | 149.1 |
| maggia_valais_20240629 | 384 | 153 | 149.9 |
| boris_20240913 | 4,644 | 1,256 | 486.2 |
| valencia_dana_20241029 | 308 | 154 | 149.9 |
