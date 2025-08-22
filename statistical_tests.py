import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

OUTPUT_TEMPLATE = (
    'Normality tests (D\'Agostino-Pearson):\n'
    'GME volatility: {gme_norm:.3g}\n'
    'OPEN volatility: {open_norm:.3g}\n'
    'Equal variance test (Levene): {levene_p:.3g}\n'
    'Volatility comparison (Mann-Whitney): {vol_mann:.3g}\n'
    'Mentions comparison (Mann-Whitney): {mention_mann:.3g}\n'
    'Correlations (Spearman):\n'
    'GME mentions-volatility: r={gme_corr:.3f}, p={gme_corr_p:.3g}\n'
    'OPEN mentions-volatility: r={open_corr:.3f}, p={open_corr_p:.3g}\n'
    'Activity levels ANOVA: F={anova_f:.3f}, p={anova_p:.3g}'
)

def main():
    os.makedirs('stats', exist_ok=True)
    
    data = pd.read_csv('merge_cleaned/dataset.csv')
    gme = data[data['ticker'] == 'GME']
    open_data = data[data['ticker'] == 'OPEN']
    
    # normality test
    # Is volatility data normally distributed? 
    # H0: Data follows a normal distribution
    # H1: Data does not follow a normal distribution
    gme_norm = stats.normaltest(gme['vol_5d'].dropna()).pvalue
    open_norm = stats.normaltest(open_data['vol_5d'].dropna()).pvalue
    # result: data is not normally distributed

    # Levene's test
    # Question: Do both stocks have similar volatility spread?
    # H0: GME and OPEN have equal variances in volatility
    # H1: GME and OPEN have different variances in volatility
    levene_p = stats.levene(gme['vol_5d'].dropna(), open_data['vol_5d'].dropna()).pvalue
    # result: data has different volatility variances

    # mann-whitney u test
    # Question: Does one stock tend to be more volatile than the other?
    # H0: GME and OPEN volatility distributions are identical
    # H1: GME and OPEN have significantly different volatility distributions
    vol_mann = stats.mannwhitneyu(gme['vol_5d'].dropna(), open_data['vol_5d'].dropna()).pvalue
    # result: GME is more volatile than OPEN

    # Question: Does one stock get mentioned significantly more than the other?
    # H0: GME and OPEN mention count distributions are identical  
    # H1: GME and OPEN have significantly different mention patterns
    mention_mann = stats.mannwhitneyu(gme['mention_count'], open_data['mention_count']).pvalue
    # result: GME is mentioned more than OPEN

    # spearman correlation
    # Question: Do more Reddit mentions predict higher volatility for each stock?
    # H0: No correlation between mentions and volatility (ρ = 0)
    # H1: There is a correlation between mentions and volatility (ρ ≠ 0)
    gme_corr, gme_corr_p = stats.spearmanr(gme['mention_count'], gme['vol_5d'])
    open_corr, open_corr_p = stats.spearmanr(open_data['mention_count'], open_data['vol_5d'])
    # result: both show significant positive correlations, but OPEN’s correlation is stronger than GME’s

    # anova one-way
    # Question: Does the level of Reddit activity predict volatility across both stocks?
    # H0: All activity levels (none/low/medium/high mentions) have equal mean volatility
    # H1: At least one activity level has significantly different mean volatility
    data['activity_level'] = pd.cut(data['mention_count'], 
                                    bins=[-0.1, 0, 5, 20, float('inf')],
                                    labels=['none', 'low', 'medium', 'high'])
    
    groups = [group['vol_5d'].dropna() for name, group in data.groupby('activity_level')]
    anova_f, anova_p = stats.f_oneway(*groups)
    # result: At least one activity level has a significantly different mean volatility

    # print results
    print(OUTPUT_TEMPLATE.format(
        gme_norm=gme_norm,
        open_norm=open_norm,
        levene_p=levene_p,
        vol_mann=vol_mann,
        mention_mann=mention_mann,
        gme_corr=gme_corr,
        gme_corr_p=gme_corr_p,
        open_corr=open_corr,
        open_corr_p=open_corr_p,
        anova_f=anova_f,
        anova_p=anova_p
    ))
    
    # save results
    results = pd.DataFrame({
        'test': ['gme_normality', 'open_normality', 'volatility_levene', 
                 'volatility_mann_whitney', 'mentions_mann_whitney', 
                 'gme_correlation', 'open_correlation', 'activity_anova'],
        'p_value': [gme_norm, open_norm, levene_p, vol_mann, mention_mann, 
                    gme_corr_p, open_corr_p, anova_p],
        'significant': [p < 0.05 for p in [gme_norm, open_norm, levene_p, 
                                           vol_mann, mention_mann, gme_corr_p, 
                                           open_corr_p, anova_p]]
    })
    results.to_csv('stats/stats_results.csv', index=False)

    # tukey post-hoc test (only if anova is significant)
    # Question: Which specific activity levels differ from each other?
    # H0: No difference between specific pairs of activity levels
    # H1: Significant difference between specific pairs 
    if anova_p < 0.05:
        tukey_data = data[['vol_5d', 'activity_level']].dropna()
        tukey = pairwise_tukeyhsd(tukey_data['vol_5d'], tukey_data['activity_level'])
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df.to_csv('stats/tukey_results.csv', index=False)

if __name__ == '__main__':
    main()