# Results

Runs: 20261007-210647-dist, 20261007-211754-sweep, 20261007-212138-sweep2, 20261007-213106-cold, 20261007-213516-conc3, 20261007-213739-quality  
Requests: 558  
Models: eleven_v4, eleven_v4_turbo

## Latency

TTFA in ms; p50 shown with bootstrap 95% CI.

| model           | conn   |   conc |   n | TTFA p50         |   TTFA p95 | TTFA p99   |   TTFA IQR |   est. server TTFA p50 |   RTF p50 | stall rate (0/100/250ms buf)   |
|:----------------|:-------|-------:|----:|:-----------------|-----------:|:-----------|-----------:|-----------------------:|----------:|:-------------------------------|
| eleven_v4       | cold   |      1 |  40 | 1634 [1620-1686] |       1877 | n<100      |        106 |                    106 |      0.73 | 2% / 0% / 0%                   |
| eleven_v4       | warm   |      1 | 219 | 1695 [1675-1719] |       1942 | 2459       |        162 |                    186 |      0.53 | 7% / 1% / 1%                   |
| eleven_v4       | warm   |      3 |  20 | 1966 [1904-2005] |       2088 | n<100      |        146 |                    425 |      0.43 | 0% / 0% / 0%                   |
| eleven_v4_turbo | cold   |      1 |  40 | 1011 [971-1047]  |       1185 | n<100      |        123 |                   -525 |      0.52 | 2% / 2% / 2%                   |
| eleven_v4_turbo | warm   |      1 | 219 | 1059 [1040-1079] |       1422 | 1860       |        183 |                   -446 |      0.32 | 1% / 0% / 0%                   |
| eleven_v4_turbo | warm   |      3 |  20 | 1289 [1107-1418] |       1504 | n<100      |        340 |                   -291 |      0.27 | 0% / 0% / 0%                   |



Mann-Whitney U (eleven_v4 vs eleven_v4_turbo, TTFA): p = 6.8e-87; P(eleven_v4 request faster) = 0.02


## Scaling with input length

| model           |   TTFA ms per 100 chars |   TTFA intercept ms |   total ms per 100 chars |   R2 total |
|:----------------|------------------------:|--------------------:|-------------------------:|-----------:|
| eleven_v4       |                     3.7 |                1759 |                     1424 |       0.99 |
| eleven_v4_turbo |                     6.7 |                1114 |                      565 |       0.98 |


![length](length_sweep.png)


![ecdf](ttfa_ecdf.png)

![timeline](stream_timeline.png)


## Cost

Measured from account credit counter before/after single-model runs; plan- and date-specific, not list pricing.


| model           |   credits / char |   credits / 1K chars |   chars / s of audio |   credits / min of audio |   TTFA p50 (ms) |
|:----------------|-----------------:|---------------------:|---------------------:|-------------------------:|----------------:|
| eleven_v4       |            0.109 |                  109 |                 15.2 |                     99.6 |            1695 |
| eleven_v4_turbo |            0.055 |                   55 |                 15.2 |                     50.2 |            1059 |



## Quality (objective)

| model           |   wer mean |   wer std |   cer mean |   cer std |   utmos mean |   utmos std |   f0_range_st mean |   f0_range_st std |   f0_std_st mean |   f0_std_st std |   energy_std_db mean |   energy_std_db std |   speech_rate_wps mean |   speech_rate_wps std |   pause_count mean |   pause_count std |   lead_silence_ms mean |   lead_silence_ms std |   spk_sim mean |   spk_sim std |
|:----------------|-----------:|----------:|-----------:|----------:|-------------:|------------:|-------------------:|------------------:|-----------------:|----------------:|---------------------:|--------------------:|-----------------------:|----------------------:|-------------------:|------------------:|-----------------------:|----------------------:|---------------:|--------------:|
| eleven_v4       |      0.064 |     0.113 |      0.035 |     0.072 |        3.743 |       0.345 |             16.919 |             2.684 |            5.374 |           0.843 |                7.92  |               0.662 |                  2.638 |                 0.553 |              0.981 |             0.623 |                 31.154 |                17.643 |          0.946 |         0.024 |
| eleven_v4_turbo |      0.067 |     0.118 |      0.037 |     0.076 |        3.802 |       0.316 |             16.934 |             2.626 |            5.361 |           0.85  |                7.977 |               0.655 |                  2.634 |                 0.552 |              1.019 |             0.682 |                 35.673 |                17.668 |          0.943 |         0.025 |


### Question intonation

final_rise_st: pitch of the last 150 ms relative to the utterance median (semitones). Positive = rising ending.


| model           |   ('final_rise_st', 'mean') |   ('final_rise_st', 'std') |   ('final_slope_st_per_s', 'mean') |   ('final_slope_st_per_s', 'std') |
|:----------------|----------------------------:|---------------------------:|-----------------------------------:|----------------------------------:|
| eleven_v4       |                       -4.79 |                       4.78 |                             -15.74 |                             13.93 |
| eleven_v4_turbo |                       -5.44 |                       5.08 |                             -17.62 |                             14.49 |


### By category

| category      |   ('f0_range_st', 'eleven_v4') |   ('f0_range_st', 'eleven_v4_turbo') |   ('utmos', 'eleven_v4') |   ('utmos', 'eleven_v4_turbo') |   ('wer', 'eleven_v4') |   ('wer', 'eleven_v4_turbo') |
|:--------------|-------------------------------:|-------------------------------------:|-------------------------:|-------------------------------:|-----------------------:|-----------------------------:|
| ack           |                         17.749 |                               17.029 |                    3.74  |                          3.673 |                  0     |                        0     |
| acronyms_urls |                         18.65  |                               18.268 |                    3.888 |                          3.974 |                  0     |                        0     |
| apology       |                          9.381 |                                9.841 |                    3.588 |                          3.55  |                  0     |                        0     |
| dates_ids     |                         15.772 |                               15.544 |                    4.265 |                          4.277 |                  0.222 |                        0.229 |
| empathy       |                         17.078 |                               16.241 |                    3.217 |                          3.268 |                  0     |                        0     |
| excited       |                         17.772 |                               18.374 |                    3.763 |                          3.929 |                  0     |                        0     |
| hindi         |                         16.582 |                               16.646 |                  nan     |                        nan     |                  0.065 |                        0.048 |
| hinglish      |                         17.656 |                               18.254 |                    3.187 |                          3.548 |                  0.309 |                        0.353 |
| info          |                         19.469 |                               19.257 |                    4.099 |                          4.054 |                  0     |                        0     |
| names_places  |                         17.051 |                               17.1   |                    3.717 |                          3.829 |                  0     |                        0     |
| numbers       |                         14.885 |                               15.256 |                    4.11  |                          4.196 |                  0.238 |                        0.238 |
| question      |                         17.917 |                               17.852 |                    3.64  |                          3.597 |                  0     |                        0     |
| question_open |                         19.99  |                               20.479 |                    3.703 |                          3.732 |                  0     |                        0     |


### Paired comparison (same input text)

| metric          |   mean(eleven_v4 - eleven_v4_turbo) | 95% CI          |   items |   wilcoxon p |
|:----------------|------------------------------------:|:----------------|--------:|-------------:|
| wer             |                              -0.003 | [-0.010, 0.003] |      13 |        0.75  |
| cer             |                              -0.002 | [-0.008, 0.002] |      13 |        1     |
| utmos           |                              -0.059 | [-0.125, 0.000] |      12 |        0.129 |
| f0_range_st     |                              -0.015 | [-0.256, 0.249] |      13 |        0.946 |
| final_rise_st   |                               0.427 | [-0.145, 1.009] |      13 |        0.168 |
| speech_rate_wps |                               0.004 | [-0.002, 0.010] |      13 |        0.191 |
| spk_sim         |                               0.003 | [-0.000, 0.006] |      13 |        0.191 |