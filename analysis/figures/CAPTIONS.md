# Figure captions

Written by the analysis scripts; numbers are regenerated with each run.

## step1_sir_fits

Observed daily cases per 100k (centered 7-day moving average, blue) and the least-squares SIR fit (dashed orange; 1/gamma = 7 days, fitted transmission rate, initial infected fraction and reporting fraction) for the first major pre-2021 wave in five example counties, shown with 21 days either side of the fitted wave. Shading: 14-day window of the Poisson growth-rate fit. Titles give the SIR-implied final attack rate (share of the population ever infected if the wave ran to completion: Los Angeles 59%, Maricopa 51%, Miami-Dade 64%, Hidalgo 74%, Lancaster 40%) and R_eff from the SIR fit and from the growth rate (1 + r/gamma). In Maricopa the smoothed series drops to zero in late December and then spikes: a holiday reporting gap (no reports over the holidays, then a catch-up), not a real change in incidence.

## step2_reff_histogram

Distribution of growth-rate R_eff (1 + r/gamma, 1/gamma = 7 days, r from a Poisson GLM on daily counts with day-of-week effects over a 14-day window) at the onset of the first major pre-2021 wave, 915 counties with population ≥ 50,000. Dashed line: median (1.32). Dotted line: R_eff = 1, the growth threshold. The x-axis is cut at 0 and 4; 5 counties fall outside (1 below 0, 4 above 4). Values below 0 are noisy estimates of near-zero growth: when r is close to zero its sampling error can push 1 + r/gamma below 0, which has no physical meaning.

## step2_reff_vs_onset

Growth-rate R_eff (1 + r/gamma, 1/gamma = 7 days) at wave onset against onset date, for every standard-preset wave with onset in 2020 (1688 waves: 894 first major waves in blue, 794 other waves in gray). Fits with SE(R_eff) > 1 are treated as failed and excluded (53 waves). Orange line: monthly median of all waves, plotted at mid-month; shaded band: monthly interquartile range. Dotted line: R_eff = 1. The y-axis is cut at 0 and 4; 3 waves fall outside (2 below 0, 1 above 4).

## step3_factor_scatter_month_adjusted

Month-adjusted R_eff (residual from a WLS model of R_eff on onset-month fixed effects, weights 1/SE²) against each of the six county factors, 915 counties. Point area is proportional to the weight; line: WLS fit of the residual on the raw factor. Panel titles give the month-adjusted WLS result for that factor from the Step 3 main table: change in R_eff per 1 SD of the factor (R_eff ~ z(factor) + onset-month fixed effects, weights 1/SE²) and its p-value with SEs clustered by state. Bonferroni threshold for six tests: p < 0.0083. y-axis: central 99% of values.

## step4_dpc_vs_age65_month_adjusted

Deaths per case against % of residents aged 65+, adjusted for wave onset month: each county's observed deaths divided by the deaths expected from a Poisson model with onset-month effects and a log(cases) offset (915 counties). Point area is proportional to cases in the wave. Orange line: Poisson fit of observed deaths on % aged 65+ with log(expected) as offset, drawn only from the 1st to the 99th percentile of % aged 65+ (10.2% to 32.7%). Diamonds: pooled observed / expected (sum of deaths / sum of expected) in 10 equal-count bins of % aged 65+, plotted at each bin's mean, with 95% percentile CIs from 1,000 bootstrap resamples of counties within the bin. Dotted line: observed = expected. The y-axis stops at the 99th percentile; 7 counties lie above it.

## step4_forest_reff_vs_dpc

Association of each county factor with growth-rate R_eff (left; change in R_eff per 1 SD, WLS with weights 1/SE²) and with deaths per case in the same wave (right; % change per 1 SD, quasi-Poisson with log(cases) offset). Bars: Bonferroni-adjusted 99.2% CIs (6 tests). Filled points: p < 0.0083. Each factor estimated in a separate model with onset-month fixed effects; SEs clustered by state.

## step4_pooled_dpc_by_month

Pooled deaths per case (sum of deaths over [onset + 14, end + 14] days / sum of cases over [onset, end]) for counties' first major pre-2021 wave, by wave onset month; n = counties per month. Error bars: 95% percentile CIs from 1,000 bootstrap resamples of counties within each month. Dashed line: pooled value over all 915 counties (1.76%).
