from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

STOPWORDS = {
    'a','an','and','are','as','at','be','by','for','from','how','in','is','it','its','of','on','or','that','the','this','to','with',
    'de','del','la','el','los','las','para','por','con','sin','y','en','un','una','unos','unas','o',
    'le','les','des','du','au','aux','et','pour','sur','dans','sans',
    'der','die','das','und','mit','ohne','ein','eine','einer','eines','zu','im','am','von','den','dem','des'
}

COLUMN_ALIASES = {
    'search_term': ['Customer search term', 'Search term', 'Customer Search Term', 'search term'],
    'clicks': ['Clicks', 'clicks'],
    'impressions': ['Impressions', 'impressions'],
    'spend': ['Total cost (USD)', 'Spend', 'Cost', 'Total spend', 'Spend (USD)', 'Total cost'],
    'sales': ['Sales (USD)', '7 Day Total Sales ', 'Sales', 'Attributed sales', 'Total sales', 'Revenue'],
    'orders': ['Purchases', 'Orders', '7 Day Total Orders (#)', 'Conversions', 'Attributed conversions'],
    'cpc': ['CPC (USD)', 'CPC', 'Avg CPC'],
    'ctr': ['CTR', 'ctr'],
}

NEGATIVE_ALIASES = {
    'keyword': ['Keyword', 'Negative keyword', 'Customer search term', 'Search term'],
    'match_type': ['Target match type', 'Match type', 'Keyword match type'],
}


@dataclass
class AnalyzerSettings:
    min_clicks_floor: int = 6
    low_sales_max: float = 1.0
    low_relevancy_ratio: float = 0.5
    zero_sales_quantile: float = 0.75
    low_sales_click_quantile: float = 0.70
    high_spend_quantile: float = 0.70
    pair_min_clicks_floor: int = 4
    exclude_numeric_tokens: bool = True
    min_token_length: int = 2


def normalize_col(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(name).strip().lower()).strip()


def find_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    norm_to_actual = {normalize_col(c): c for c in df.columns}
    for alias in aliases:
        normalized = normalize_col(alias)
        if normalized in norm_to_actual:
            return norm_to_actual[normalized]
    return None


def detect_columns(df: pd.DataFrame, alias_map: Dict[str, Sequence[str]]) -> Dict[str, Optional[str]]:
    return {k: find_column(df, v) for k, v in alias_map.items()}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('$', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors='coerce').fillna(0)


def load_search_term_file_objects(files: Iterable[BinaryIO]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file)
        cols = detect_columns(df, COLUMN_ALIASES)
        required = ['search_term', 'clicks', 'spend', 'sales', 'orders']
        missing = [r for r in required if not cols.get(r)]
        if missing:
            raise ValueError(f"{getattr(file, 'name', 'uploaded file')} is missing required columns: {missing}")

        renamed = pd.DataFrame({
            'source_file': getattr(file, 'name', 'uploaded_file.csv'),
            'search_term': df[cols['search_term']].astype(str),
            'clicks': _coerce_numeric(df[cols['clicks']]),
            'spend': _coerce_numeric(df[cols['spend']]),
            'sales': _coerce_numeric(df[cols['sales']]),
            'orders': _coerce_numeric(df[cols['orders']]),
            'impressions': _coerce_numeric(df[cols['impressions']]) if cols.get('impressions') else 0,
            'cpc': _coerce_numeric(df[cols['cpc']]) if cols.get('cpc') else 0.0,
            'ctr': _coerce_numeric(df[cols['ctr']]) if cols.get('ctr') else 0.0,
        })
        frames.append(renamed)

    if not frames:
        raise ValueError('No search term files were uploaded.')

    combined = pd.concat(frames, ignore_index=True)
    combined['search_term'] = combined['search_term'].fillna('').str.strip().str.lower()
    combined = combined[combined['search_term'] != '']
    agg = combined.groupby('search_term', as_index=False).agg({
        'clicks': 'sum', 'spend': 'sum', 'sales': 'sum', 'orders': 'sum', 'impressions': 'sum'
    })
    agg['cvr'] = agg['orders'] / agg['clicks'].replace(0, pd.NA)
    agg['acos'] = agg['spend'] / agg['sales'].replace(0, pd.NA)
    agg['cpc'] = agg['spend'] / agg['clicks'].replace(0, pd.NA)
    return agg.fillna(0)


def load_negative_keyword_file(file: Optional[BinaryIO]) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=['keyword', 'match_type'])

    df = pd.read_csv(file)
    cols = detect_columns(df, NEGATIVE_ALIASES)
    if not cols.get('keyword'):
        raise ValueError(f"{getattr(file, 'name', 'negative keywords file')} is missing a keyword column")

    out = pd.DataFrame({
        'keyword': df[cols['keyword']].astype(str).str.strip().str.lower(),
        'match_type': df[cols['match_type']].astype(str).str.strip() if cols.get('match_type') else ''
    })
    out = out[out['keyword'] != '']
    return out.drop_duplicates()


def tokenize(text: str, settings: AnalyzerSettings) -> List[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", str(text).lower())
    clean: List[str] = []
    for token in tokens:
        if settings.exclude_numeric_tokens and token.isdigit():
            continue
        if len(token) < settings.min_token_length:
            continue
        if token in STOPWORDS:
            continue
        clean.append(token)
    return clean


def build_ngram_rows(search_df: pd.DataFrame, settings: AnalyzerSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    single_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for _, row in search_df.iterrows():
        tokens = tokenize(row['search_term'], settings)
        singles = sorted(set(tokens))
        pairs = sorted(
            set(' '.join(tokens[i:i + 2]) for i in range(len(tokens) - 1) if len(tokens[i:i + 2]) == 2)
        )
        metrics = {
            'search_term': row['search_term'],
            'clicks': row['clicks'],
            'spend': row['spend'],
            'sales': row['sales'],
            'orders': row['orders'],
            'term_cvr': row['cvr'],
        }
        for single in singles:
            entry = metrics.copy()
            entry['token'] = single
            single_rows.append(entry)
        for pair in pairs:
            entry = metrics.copy()
            entry['token'] = pair
            pair_rows.append(entry)

    return pd.DataFrame(single_rows), pd.DataFrame(pair_rows)


def build_campaign_baseline(search_df: pd.DataFrame) -> Dict[str, float]:
    clicks = float(search_df['clicks'].sum())
    sales = float(search_df['sales'].sum())
    orders = float(search_df['orders'].sum())
    spend = float(search_df['spend'].sum())
    return {
        'clicks': clicks,
        'sales': sales,
        'orders': orders,
        'spend': spend,
        'cvr': orders / clicks if clicks else 0.0,
        'avg_cpc': spend / clicks if clicks else 0.0,
        'avg_sale_per_term': float(search_df['sales'].mean()) if len(search_df) else 0.0,
        'clicks_p90': float(search_df['clicks'].quantile(0.90)) if len(search_df) else 1.0,
    }


def classify_recommendation(row: pd.Series) -> str:
    if row['already_negative']:
        return 'Already negative'
    if row['issue_zero_sales'] and row['issue_high_spend']:
        return 'Strong negative candidate'
    if row['issue_zero_sales'] or row['issue_low_relevancy']:
        return 'Review as negative'
    return 'Monitor'


def build_reason(row: pd.Series) -> str:
    reasons: List[str] = []
    if row['issue_zero_sales']:
        reasons.append('high clicks with zero sales')
    if row['issue_low_relevancy']:
        reasons.append('conversion rate far below campaign baseline')
    if row['issue_high_spend']:
        reasons.append('high spend without sales')
    if not reasons:
        reasons.append('traffic present but evidence is weaker')
    return '; '.join(reasons)


def aggregate_token_stats(
    token_rows: pd.DataFrame,
    token_type: str,
    baseline: Dict[str, float],
    negatives: pd.DataFrame,
    settings: AnalyzerSettings,
) -> pd.DataFrame:
    if token_rows.empty:
        return pd.DataFrame(columns=[
            'token', 'search_terms', 'clicks', 'spend', 'sales', 'orders', 'cvr', 'acos', 'cpc',
            'token_type', 'already_negative', 'click_share', 'sales_share', 'relative_cvr',
            'issue_zero_sales', 'issue_low_relevancy', 'issue_high_spend', 'score', 'recommendation', 'reason'
        ])

    neg_set = set(negatives['keyword'].tolist()) if not negatives.empty else set()
    grouped = token_rows.groupby('token', as_index=False).agg(
        search_terms=('search_term', 'nunique'),
        clicks=('clicks', 'sum'),
        spend=('spend', 'sum'),
        sales=('sales', 'sum'),
        orders=('orders', 'sum'),
    )
    grouped['cvr'] = grouped['orders'] / grouped['clicks'].replace(0, pd.NA)
    grouped['acos'] = grouped['spend'] / grouped['sales'].replace(0, pd.NA)
    grouped['cpc'] = grouped['spend'] / grouped['clicks'].replace(0, pd.NA)
    grouped = grouped.fillna(0)
    grouped['token_type'] = token_type
    grouped['already_negative'] = grouped['token'].isin(neg_set)
    grouped['click_share'] = grouped['clicks'] / max(baseline['clicks'], 1)
    grouped['sales_share'] = grouped['sales'] / max(baseline['sales'], 1e-9)
    grouped['relative_cvr'] = grouped['cvr'] / max(baseline['cvr'], 1e-9)

    zero_sales_click_threshold = max(
        settings.min_clicks_floor if token_type == 'single_word' else settings.pair_min_clicks_floor,
        grouped.loc[grouped['sales'] == 0, 'clicks'].quantile(settings.zero_sales_quantile) if (grouped['sales'] == 0).any() else 0,
    )
    low_sales_click_threshold = max(
        settings.min_clicks_floor if token_type == 'single_word' else settings.pair_min_clicks_floor,
        grouped['clicks'].quantile(settings.low_sales_click_quantile),
    )
    spend_threshold = max(baseline['avg_cpc'] * 3, grouped['spend'].quantile(settings.high_spend_quantile))

    grouped['issue_zero_sales'] = (grouped['sales'] <= 0) & (grouped['clicks'] >= zero_sales_click_threshold)
    grouped['issue_low_relevancy'] = (
        (grouped['clicks'] >= low_sales_click_threshold)
        & (grouped['relative_cvr'] < settings.low_relevancy_ratio)
        & (grouped['sales'] <= settings.low_sales_max)
    )
    grouped['issue_high_spend'] = (grouped['sales'] <= 0) & (grouped['spend'] >= spend_threshold)

    grouped['score'] = (
        grouped['issue_zero_sales'].astype(int) * 4
        + grouped['issue_low_relevancy'].astype(int) * 3
        + grouped['issue_high_spend'].astype(int) * 3
        + (grouped['clicks'] / max(baseline['clicks_p90'], 1)).clip(upper=2) * 2
        + (1 - grouped['relative_cvr'].clip(upper=1.5)).clip(lower=0) * 2
    ).round(2)

    grouped['recommendation'] = grouped.apply(classify_recommendation, axis=1)
    grouped['reason'] = grouped.apply(build_reason, axis=1)
    return grouped.sort_values(['score', 'clicks', 'spend'], ascending=[False, False, False])


def analyze_uploaded_files(
    search_term_files: Sequence[BinaryIO],
    negative_keyword_file: Optional[BinaryIO],
    settings: Optional[AnalyzerSettings] = None,
) -> Dict[str, object]:
    active_settings = settings or AnalyzerSettings()
    search_df = load_search_term_file_objects(search_term_files)
    negatives = load_negative_keyword_file(negative_keyword_file)
    baseline = build_campaign_baseline(search_df)
    singles_raw, pairs_raw = build_ngram_rows(search_df, active_settings)
    single_stats = aggregate_token_stats(singles_raw, 'single_word', baseline, negatives, active_settings)
    pair_stats = aggregate_token_stats(pairs_raw, 'word_pair', baseline, negatives, active_settings)

    recommended = pd.concat([single_stats, pair_stats], ignore_index=True)
    recommended = recommended[recommended['recommendation'].isin(['Strong negative candidate', 'Review as negative'])]
    recommended = recommended[~recommended['already_negative']]
    recommended = recommended.sort_values(['score', 'clicks', 'spend'], ascending=[False, False, False])

    min_term_clicks = max(active_settings.min_clicks_floor, search_df['clicks'].quantile(active_settings.low_sales_click_quantile))
    top_terms = search_df[(search_df['clicks'] >= min_term_clicks) & (search_df['sales'] <= 0)].sort_values(
        ['clicks', 'spend'], ascending=[False, False]
    )

    summary = {
        'campaign_baseline': baseline,
        'rows_search_terms': int(len(search_df)),
        'rows_negative_keywords': int(len(negatives)),
        'recommended_candidates': int(len(recommended)),
        'top_recommended_preview': recommended.head(20).to_dict(orient='records'),
        'settings': active_settings.__dict__,
    }

    return {
        'summary': summary,
        'search_terms': search_df,
        'negative_keywords': negatives,
        'single_words': single_stats,
        'word_pairs': pair_stats,
        'recommendations': recommended,
        'high_click_zero_sale_terms': top_terms,
    }


def build_download_zip(results: Dict[str, object]) -> bytes:
    output_map = {
        '00_summary.json': json.dumps(results['summary'], indent=2),
        '01_cleaned_search_terms.csv': results['search_terms'].to_csv(index=False),
        '02_negative_keywords_cleaned.csv': results['negative_keywords'].to_csv(index=False),
        '03_single_word_analysis.csv': results['single_words'].to_csv(index=False),
        '04_word_pair_analysis.csv': results['word_pairs'].to_csv(index=False),
        '05_recommended_negative_candidates.csv': results['recommendations'].to_csv(index=False),
        '06_high_click_zero_sale_search_terms.csv': results['high_click_zero_sale_terms'].to_csv(index=False),
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, content in output_map.items():
            zf.writestr(filename, content)
    buffer.seek(0)
    return buffer.getvalue()


def save_results_to_folder(results: Dict[str, object], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / '00_summary.json').write_text(json.dumps(results['summary'], indent=2), encoding='utf-8')
    results['search_terms'].to_csv(outdir / '01_cleaned_search_terms.csv', index=False)
    results['negative_keywords'].to_csv(outdir / '02_negative_keywords_cleaned.csv', index=False)
    results['single_words'].to_csv(outdir / '03_single_word_analysis.csv', index=False)
    results['word_pairs'].to_csv(outdir / '04_word_pair_analysis.csv', index=False)
    results['recommendations'].to_csv(outdir / '05_recommended_negative_candidates.csv', index=False)
    results['high_click_zero_sale_terms'].to_csv(outdir / '06_high_click_zero_sale_search_terms.csv', index=False)
