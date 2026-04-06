# Bot vs Human Notes: Analysis Report (Complete Raters Only)

Complete raters: 13,721 ratings, 1,674 notes


## %CRH and %CRNH (complete-raters noteParams)

```
internalRatingStatus  CURRENTLY_RATED_HELPFUL  CURRENTLY_RATED_NOT_HELPFUL  NEED_MORE_RATINGS   All
writer                                                                                             
bot                                        16                            6                645   667
human                                      19                           14                974  1007
All                                        35                           20               1619  1674
```

  %CRH: LLM=2.40%  Human=1.89%
  %CRNH: LLM=0.90%  Human=1.39%

--- Note intercept (internalNoteIntercept, complete raters) ---
  n_bot=315, n_human=458
  mean: Bot=0.21  Human=0.18
  median: Bot=0.22  Human=0.20
  CRH LMM (Complete raters): AI coef=0.0015, SE=0.0044, z=0.3315, p=0.740271 (unadjusted), p_adj=0.740271 (BH) (n=1,674)
  CRNH LMM (Complete raters): AI coef=-0.0049, SE=0.0053, z=-0.9261, p=0.354371 (unadjusted), p_adj=0.531557 (BH) (n=1,674)
  Note intercept LMM (Complete raters): AI coef=0.0188, SE=0.0064, z=2.9442, p=0.003238 (unadjusted), p_adj=0.009713 (BH) (n=773)