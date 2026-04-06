# Bot vs Human Notes: Analysis Report

Run rating-level mixed effects analysis with data/ratings_analysis_df.csv in analysis.R
## Analysis by rater bucket and writer

Mean % Helpful by writer and rater bucket (95% CI across notes):

Bucket = left
     bot: mean= 79.55%  (n=1300, CI=[77.83, 81.27])
   human: mean= 68.10%  (n=1183, CI=[65.95, 70.24])

Bucket = neutral
     bot: mean= 89.58%  (n=1174, CI=[88.32, 90.84])
   human: mean= 76.83%  (n=1049, CI=[74.86, 78.79])

Bucket = right
     bot: mean= 73.13%  (n=1300, CI=[71.32, 74.93])
   human: mean= 67.46%  (n=1196, CI=[65.52, 69.41])
Mean % Unhelpful by writer and rater bucket (95% CI across notes):

Bucket = left
     bot: mean= 17.98%  (n=1300, CI=[16.33, 19.63])
   human: mean= 28.72%  (n=1183, CI=[26.61, 30.83])

Bucket = neutral
     bot: mean=  7.60%  (n=1174, CI=[6.52, 8.69])
   human: mean= 18.40%  (n=1049, CI=[16.56, 20.25])

Bucket = right
     bot: mean= 23.65%  (n=1300, CI=[21.92, 25.38])
   human: mean= 28.58%  (n=1196, CI=[26.68, 30.48])

Saved bar chart: outputs/rating_analysis_bot_vs_human_barchart.png

## Note-level analysis: bot vs human

Final rating status distribution by writer:
```
finalRatingStatus  CURRENTLY_RATED_HELPFUL  CURRENTLY_RATED_NOT_HELPFUL  NEEDS_MORE_RATINGS   All
writer                                                                                           
bot                                    211                           18                1385  1614
human                                  240                           55                1037  1332
All                                    451                           73                2422  2946
```
  %CRH: Bot=13.07%  Human=18.02%
  %CRNH: Bot=1.12%  Human=4.13%
  CRH LMM (Full sample): AI coef=-0.0583, SE=0.0128, z=-4.5566, p=0.000005 (n=2,946)
  CRNH LMM (Full sample): AI coef=-0.0270, SE=0.0057, z=-4.7148, p=0.000002 (n=2,946)

--- coreNoteIntercept ---
  n_bot=1243, n_human=1130
  mean: Bot=0.25  Human=0.24
  median: Bot=0.25  Human=0.25
  Note intercept LMM (Full sample): AI coef=0.0069, SE=0.0061, z=1.1429, p=0.253071 (n=2,373)

--- numRatings ---
  n_bot=1614, n_human=1332
  mean: Bot=59.50  Human=109.94
  median: Bot=22.00  Human=51.00
  Mann-Whitney U: U=740798.00, p=0.000000

--- Time to CRH (hours) among CRH notes ---
  n_bot=211, n_human=240
  mean: Bot=21.96  Human=17.89
  median: Bot=7.38  Human=5.90
  Mann-Whitney U: U=27979.00, p=0.054240

### Subset: notes with numRatings >= 30 (exploratory, unadjusted)
Notes with numRatings >= 30: 1,538 (bot: 680, human: 858)

Final rating status distribution:
```
finalRatingStatus  CURRENTLY_RATED_HELPFUL  CURRENTLY_RATED_NOT_HELPFUL  NEEDS_MORE_RATINGS   All
writer                                                                                           
bot                                    190                            4                 486   680
human                                  232                           39                 587   858
All                                    422                           43                1073  1538
```
  %CRH: Bot=27.94%  Human=27.04%
  %CRNH: Bot=0.59%  Human=4.55%
  CRH LMM (numRatings>=30): AI coef=-0.0062, SE=0.0196, z=-0.3169, p=0.751292 (n=1,538)
  CRNH LMM (numRatings>=30): AI coef=-0.0346, SE=0.0084, z=-4.1286, p=0.000037 (n=1,538)

--- coreNoteIntercept ---
  n_bot=645, n_human=798
  mean: Bot=0.31  Human=0.28
  median: Bot=0.31  Human=0.30
  Note intercept LMM (numRatings>=30): AI coef=0.0185, SE=0.0077, z=2.4100, p=0.015951 (n=1,443)

--- numRatings ---
  n_bot=680, n_human=858
  mean: Bot=126.39  Human=163.37
  median: Bot=67.00  Human=91.50
  Mann-Whitney U: U=236530.50, p=0.000000

--- Time to CRH (hours) among CRH notes ---
  n_bot=190, n_human=232
  mean: Bot=21.87  Human=17.63
  median: Bot=7.16  Human=5.78
  Mann-Whitney U: U=24102.00, p=0.098168

## Timing-matched analysis

Window +/- 30 min: matched 102/1614 bot notes (6.3%)
## Note-level analysis: bot vs human (timing_30min)

Final rating status distribution by writer:
```
finalRatingStatus  CURRENTLY_RATED_HELPFUL  CURRENTLY_RATED_NOT_HELPFUL  NEEDS_MORE_RATINGS  All
writer                                                                                          
bot                                     22                            2                  78  102
human                                   24                            8                  84  116
All                                     46                           10                 162  218
```
  %CRH: Bot=21.57%  Human=20.69%
  %CRNH: Bot=1.96%  Human=6.90%
  CRH LMM (timing_30min): AI coef=0.0039, SE=0.0431, z=0.0895, p=0.928668 (n=218)
  CRNH LMM (timing_30min): AI coef=-0.0494, SE=0.0284, z=-1.7406, p=0.081759 (n=218)

--- coreNoteIntercept ---
  n_bot=96, n_human=109
  mean: Bot=0.28  Human=0.24
  median: Bot=0.27  Human=0.25
  Note intercept LMM (timing_30min): AI coef=0.0412, SE=0.0193, z=2.1307, p=0.033117 (n=205)

--- numRatings ---
  n_bot=102, n_human=116
  mean: Bot=168.70  Human=132.79
  median: Bot=64.00  Human=55.00
  Mann-Whitney U: U=6491.50, p=0.215912

--- Time to CRH (hours) among CRH notes ---
  n_bot=22, n_human=24
  mean: Bot=35.69  Human=8.53
  median: Bot=6.01  Human=5.67
  Mann-Whitney U: U=288.00, p=0.605321

Window +/- 60 min: matched 190/1614 bot notes (11.8%)
## Note-level analysis: bot vs human (timing_60min)

Final rating status distribution by writer:
```
finalRatingStatus  CURRENTLY_RATED_HELPFUL  CURRENTLY_RATED_NOT_HELPFUL  NEEDS_MORE_RATINGS  All
writer                                                                                          
bot                                     41                            4                 145  190
human                                   49                           13                 153  215
All                                     90                           17                 298  405
```
  %CRH: Bot=21.58%  Human=22.79%
  %CRNH: Bot=2.11%  Human=6.05%
  CRH LMM (timing_60min): AI coef=-0.0192, SE=0.0315, z=-0.6101, p=0.541765 (n=405)
  CRNH LMM (timing_60min): AI coef=-0.0389, SE=0.0197, z=-1.9754, p=0.048219 (n=405)

--- coreNoteIntercept ---
  n_bot=178, n_human=193
  mean: Bot=0.28  Human=0.25
  median: Bot=0.28  Human=0.25
  Note intercept LMM (timing_60min): AI coef=0.0234, SE=0.0137, z=1.7030, p=0.088561 (n=371)

--- numRatings ---
  n_bot=190, n_human=215
  mean: Bot=134.92  Human=122.54
  median: Bot=54.00  Human=57.00
  Mann-Whitney U: U=21064.50, p=0.586739

--- Time to CRH (hours) among CRH notes ---
  n_bot=41, n_human=49
  mean: Bot=21.32  Human=10.21
  median: Bot=4.61  Human=5.55
  Mann-Whitney U: U=950.00, p=0.661752

Window +/- 90 min: matched 257/1614 bot notes (15.9%)
## Note-level analysis: bot vs human (timing_90min)

Final rating status distribution by writer:
```
finalRatingStatus  CURRENTLY_RATED_HELPFUL  CURRENTLY_RATED_NOT_HELPFUL  NEEDS_MORE_RATINGS  All
writer                                                                                          
bot                                     50                            4                 203  257
human                                   63                           17                 233  313
All                                    113                           21                 436  570
```
  %CRH: Bot=19.46%  Human=20.13%
  %CRNH: Bot=1.56%  Human=5.43%
  CRH LMM (timing_90min): AI coef=-0.0127, SE=0.0270, z=-0.4705, p=0.638022 (n=570)
  CRNH LMM (timing_90min): AI coef=-0.0366, SE=0.0154, z=-2.3715, p=0.017717 (n=570)

--- coreNoteIntercept ---
  n_bot=240, n_human=285
  mean: Bot=0.27  Human=0.24
  median: Bot=0.26  Human=0.24
  Note intercept LMM (timing_90min): AI coef=0.0269, SE=0.0114, z=2.3669, p=0.017936 (n=525)

--- numRatings ---
  n_bot=257, n_human=313
  mean: Bot=137.73  Human=128.11
  median: Bot=57.00  Human=59.00
  Mann-Whitney U: U=40772.00, p=0.778215

--- Time to CRH (hours) among CRH notes ---
  n_bot=50, n_human=63
  mean: Bot=20.21  Human=9.77
  median: Bot=4.96  Human=5.46
  Mann-Whitney U: U=1562.00, p=0.942396


## Human vs Bot note timing (createdAtMillis)

Tweets with both human and bot notes: 814

Human notes on tweets with both types: 1,331
  Created before first bot note: 878 (66.0%)
  Created at or after first bot note: 453 (34.0%)

### Hours earlier/later (human vs first bot note)
  Mean: 0.67 h  (median: -1.71 h)
  Std: 28.44 h
  Min: -57.84 h  Max: 547.16 h
  Percentiles: 10th=-14.55 h, 25th=-6.42 h, 75th=1.44 h, 90th=12.09 h

## Text features analysis

### Note length (word count)
  n_bot=1614, n_human=1332
  mean: Bot=35.8  Human=26.9
  median: Bot=36.0  Human=26.0
  Welch t-test: t=22.1898, p=0.000000

### Number of URLs
  n_bot=1614, n_human=1332
  mean: Bot=1.51  Human=1.23
  median: Bot=2.00  Human=1.00
  Welch t-test: t=8.6691, p=0.000000


### Source citation: Top domains by LLM vs human notes

**Top 10 domains in LLM notes:**
| Rank | Domain | % LLM notes citing | % Human notes citing |
|---|---|---|---|
| 1 | reuters.com | 7.7% | 1.6% |
| 2 | en.wikipedia.org | 7.2% | 7.6% |
| 3 | instagram.com | 5.8% | 2.0% |
| 4 | youtube.com | 5.3% | 3.9% |
| 5 | x.com | 4.2% | 18.7% |
| 6 | bbc.com | 4.2% | 1.1% |
| 7 | snopes.com | 3.9% | 0.7% |
| 8 | cnn.com | 2.4% | 0.8% |
| 9 | facebook.com | 2.0% | 0.7% |
| 10 | yahoo.com | 2.0% | 0.5% |

**Top 10 domains in human notes:**
| Rank | Domain | % LLM notes citing | % Human notes citing |
|---|---|---|---|
| 1 | x.com | 4.2% | 18.7% |
| 2 | en.wikipedia.org | 7.2% | 7.6% |
| 3 | youtube.com | 5.3% | 3.9% |
| 4 | x.com/grok | 0.4% | 2.6% |
| 5 | instagram.com | 5.8% | 2.0% |
| 6 | reuters.com | 7.7% | 1.6% |
| 7 | theguardian.com | 1.2% | 1.6% |
| 8 | t.co | 0.0% | 1.1% |
| 9 | bbc.com | 4.2% | 1.1% |
| 10 | share.google | 0.0% | 1.0% |

## CRH rate analysis: bot vs human writers
(Human writers: all from notes-00000.tsv, excluding API authors)

Bot CRH rate: 13.07% (percentile among human writers: 84.7%)

## Hit rate analysis: (#CRH - #CRNH) / total notes

Bot hit rate: 11.96% (percentile among human writers: 85.4%)
Robustness (human writers with >= 10 notes): CRH percentile=80.6%, hit rate percentile=82.5% (n=50,148 writers)
Robustness (human writers with >= 30 notes): CRH percentile=78.4%, hit rate percentile=79.0% (n=10,957 writers)

# Pair-Centric Pairwise / Bradley-Terry Analysis

Notes: 2,946 (bot: 1,614, human: 1,332)
Ratings: 108,169

## Sample sizes
  Total (rater, pair) observations: 21,978
  Unique pairs: 1,186
  Unique raters: 10,345

## Bradley-Terry (ties excluded, SEs clustered by rater)
```
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                 6371
Model:                          Logit   Df Residuals:                     6370
Method:                           MLE   Df Model:                            0
Date:                Thu, 26 Mar 2026   Pseudo R-squ.:               7.840e-11
Time:                        09:42:36   Log-Likelihood:                -4391.0
converged:                       True   LL-Null:                       -4391.0
Covariance Type:              cluster   LLR p-value:                       nan
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.1778      0.035      5.101      0.000       0.110       0.246
==============================================================================
```

beta_AI = 0.1778 (SE = 0.0349)
exp(beta_AI) = 1.1946 (odds multiplier for AI vs. human)
n (non-ties): 6,371, ties excluded: 15,607

Saved pair-centric data to outputs/pairwise_bt_comparisons.csv