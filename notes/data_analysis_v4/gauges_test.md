# v4 test split against DE / CH gauges

5,201,870 gauge-hours, 1711 stations, 486 days; weighted by the store's inclusion weights. 'coarse' = the 25 km input cell holding the gauge.

| network | field | gauge-hours | mean | P(0) | q50 | q90 | q99 | q99.9 | q99.99 | P(>=1) | P(>=5) | P(>=20) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dwd | gauge | 4275885 | 0.1144 | 0.8946 | 0 | 0.12 | 2.47 | 8.59 | 20.98 | 0.03162 | 0.002976 | 0.0001179 |
| dwd | target | 4275885 | 0.09584 | 0.8665 | 0 | 0.17 | 1.865 | 6.195 | 15.83 | 0.02313 | 0.001618 | 5.074e-05 |
| dwd | coarse | 4275885 | 0.09495 | 0.8549 | 0 | 0.2033 | 1.721 | 4.749 | 9.143 | 0.02274 | 0.0008539 | 1.506e-07 |
| smn | gauge | 925985 | 0.1699 | 0.8484 | 0 | 0.3 | 3.5 | 9.4 | 21.3 | 0.05081 | 0.005009 | 0.0001272 |
| smn | target | 925985 | 0.1279 | 0.8216 | 0 | 0.325 | 2.117 | 6.396 | 16.14 | 0.03223 | 0.001775 | 4.936e-05 |
| smn | coarse | 925985 | 0.1293 | 0.7985 | 0 | 0.3585 | 2.004 | 4.816 | 8.665 | 0.03205 | 0.0008934 | 1.204e-06 |

- dwd: on 1530 gauge-hours >= 20 mm, median target / gauge 0.43, median coarse / gauge 0.17
- smn: on 309 gauge-hours >= 20 mm, median target / gauge 0.40, median coarse / gauge 0.17
