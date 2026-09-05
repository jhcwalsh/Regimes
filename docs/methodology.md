# Regimes: methodology summary

A plain-language account of the nonparametric regime model in Mulliner, Harvey, Xia,
Fang and Van Hemert (2026) [1] and of how this replication follows it. Page numbers
refer to the published article; exhibit and equation numbers are the paper's own.

## The idea

Rather than naming regimes in advance, measure how far today's market conditions sit
from every past month across seven variables at once, and call the closest months the
current regime. Nothing is estimated or optimised; the method "is largely nonparametric"
and relies only on Z-scores and a distance [1, p. 7]. It builds on Kritzman, Page and
Turkington's turbulence index [3] and Kaya, Lee and Pornrojnangkool's nonparametric
regimes [2].

## Step 1: seven state variables

| # | Variable | Paper | This replication |
|---|---|---|---|
| 1 | Equity market | S&P 500 level, in logs | Yahoo `^GSPC` daily from 1927, month-end close |
| 2 | Yield curve | 10-year minus 3-month yield | FRED `GS10` − `TB3MS` |
| 3 | Oil | WTI crude price | FRED `WTISPLC`, monthly spot from 1946 |
| 4 | Copper | Copper price (adjusted futures) | World Bank Pink Sheet (1960–) spliced with FRED `PCOPPUSDM` |
| 5 | Monetary policy | US 3-month T-bill yield | FRED `TB3MS` |
| 6 | Volatility | VIX, prepended with realised vol before 1990 | FRED `VIXCLS`; realised vol from daily `^GSPC` |
| 7 | Stock-bond correlation | Rolling 3-year correlation of daily returns | `^GSPC` returns vs. minus the change in `^TNX`, 756-day window |

Paper sources are FRED, Man Group and Bloomberg (Exhibit 2) [1, p. 10]. Volatility and
the correlation are computed on daily data and mapped to monthly [1, p. 10].

## Step 2: transformation

Take the 12-month change, divide by the standard deviation of those changes over the
trailing ten years, winsorise at ±3 [1, pp. 10–11]:

    z_t = clip( (x_t − x_{t−12}) / σ_10y(x_t − x_{t−12}), −3, 3 )

The paper stresses this is "similar to a Z-score" but not demeaned; means come out near
zero and standard deviations near one (Exhibit 4) [1, pp. 11–13]. The 12-month change
induces one-month autocorrelation near 0.9 that fades by twelve months [1, p. 11].
Cross-correlations are low; the largest is oil vs. copper at 0.33 (Exhibit 5).

## Step 3: distance and the global score

For a target month T and each earlier month i (eq. 1) [1, p. 11]:

    d_{T,i} = sqrt( Σ_{v=1..7} (x_{i,v} − x_{T,v})² )

Lower is more similar; the score at T is zero. Variables are equally weighted, a choice
the authors flag [1, p. 7]. Euclidean distance was chosen for simplicity over
Mahalanobis [1, pp. 13–14]. Two rules shape the candidate set:

- Only months up to T count: the score is computed "for every month up to month T" [1, p. 11].
- The 36 months before T are masked, "as this helps us avoid loading up on momentum" [1, p. 14].

## Step 4: regimes and anti-regimes

Rank by score. The lowest slice is the similar regime: 15 % in the exhibits [1, p. 14],
quintiles (20 %) in the backtest, which this app adopts. The highest slice is the
anti-regime; trading in the direction of returns after those months does worst of all
quintiles [1, p. 19]. Worked examples: January 2009 selects every prior recession
including the early-1980s double dip (Exhibit 6); February and April 2020 find no
convincing pattern (Exhibit 7); August 2022 lands on 1977–80, the late 1960s and 1972–74
(Exhibit 8) [1, pp. 14–16].

## Step 5: regime shift

At each month T, average the distances from T to every earlier month with exponentially
decaying weights, most weight on the recent past: C_T = ewma{d_t : t = 0…T}. A large
value means the environment has moved a lot (Exhibit 9) [1, pp. 16–18].

    C_T = Σ_{t≤T} β^{T−t} d_{T,t} / Σ_{t≤T} β^{T−t},   β = 1 − 1/n (eq. 2),   t_½ = −ln 2 / ln β (eq. 3)

| Lookback | n | β | Half-life (months) | Paper |
|---|---|---|---|---|
| 1 year | 12 | 0.9167 | 8.0 | 8 |
| 2 years | 24 | 0.9583 | 16.3 | 16 |
| 3 years | 36 | 0.9722 | 24.6 | 25 |
| 4 years | 48 | 0.9792 | 32.9 | 33 |

Labelled peaks: Oct 82, May 83, Jul 90, Dec 90, Jul 07, Oct 08, Jan 09, Feb 20, May 20,
Oct 22, May 23 [1, p. 17].

## Step 6: prediction (the app's Factors view)

Six long-short factors (Fama-French five plus 12-month momentum). Long a factor if the
average one-month-ahead return after the similar months is positive, else short;
equal-weight the six [1, pp. 18–19]. Performance 1985–2024 (Exhibit 10) [1, p. 18]:

| Portfolio | Sharpe | Corr. to long-only |
|---|---|---|
| Quintile 1 (most similar) | 0.95 | 0.76 |
| Quintile 5 (most dissimilar) | 0.17 | 0.48 |
| Long-only | 1.00 | — |
| Q1 minus Q5 | 0.82 | 0.37 |

The spread's alpha is three standard errors from zero; positive in 80 % of years at a
15 % vol target (Exhibit 1). Robust across quantile choices (Sharpe 0.46–0.82, Exhibit 12)
and Z-score lookbacks of 1, 3 and 5 years (Exhibit 13) [1, pp. 20–21].

## Where the replication departs

| Aspect | Paper | Replication |
|---|---|---|
| Sources | FRED, Man Group, Bloomberg | FRED, Yahoo Finance and the World Bank Pink Sheet, all free |
| History | Scores from 1966 | Scores from December 1970 (daily 10-year yields start 1962) |
| Similar set | 15 % exhibits / quintiles backtest | 20 % throughout |
| Volatility splice | VIX prepended with realised vol | Same, no level adjustment at 1990 |
| Latest month | — | Lagging FRED series forward-filled up to a month |
| Factor timing | Exhibits 10–13 | Implemented (Factors view): Q1 Sharpe 0.87 vs 0.95, same 0.76 correlation; Q5 0.44 vs 0.17; spread 0.34 vs 0.82 |

The engine's tests pin the square-rooted distance, the 36-month mask, the up-to-T rule
and the regime-shift construction (`tests/`).

## Using it

A lookup, not a forecast. The similar set is small and clustered. Unprecedented months
return the least-bad matches rather than "none". Most of the paper's edge comes from the
anti-regime side. Treat subsequent-returns statistics as priors with wide error bars and
trust only conclusions that survive the paper's own robustness checks.

## References

1. Mulliner, A., Harvey, C. R., Xia, C., Fang, E. and Van Hemert, O. (2026). *Regimes.*
   The Journal of Portfolio Management, 52(4), 6–25. SSRN 5164863;
   https://people.duke.edu/~charvey/Research/Published_Papers/P176_Regimes.pdf
2. Kaya, H., Lee, W. and Pornrojnangkool, B. (2010). *Regimes: Nonparametric Identification
   and Forecasting.* The Journal of Portfolio Management, 36(2), 94–105.
3. Kritzman, M., Page, S. and Turkington, D. (2012). *Regime Shifts: Implications for
   Dynamic Strategies.* Financial Analysts Journal, 68(3), 22–39.
4. Kritzman, M., Kulasekaran, C. and Turkington, D. (2023). *Portfolio Construction When
   Regimes Are Ambiguous.* The Journal of Portfolio Management, 50(1), 8–18.
5. FRED series GS10, TB3MS, WTISPLC, PCOPPUSDM, VIXCLS; Yahoo Finance ^GSPC, ^TNX; Kenneth R. French
   Data Library (Fama-French 5 factors 2x3, Momentum factor), https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html; World Bank
   Commodity Price Data (Pink Sheet), monthly prices, https://www.worldbank.org/en/research/commodity-markets.
6. Design spec: `docs/superpowers/specs/2026-09-03-regimes-site-design.md`.
