from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from analyzer_core import AnalyzerSettings, analyze_uploaded_files, build_download_zip


def main() -> None:
    st.set_page_config(page_title='Amazon PPC Negative Keyword Analyzer', page_icon='📉', layout='wide')

    st.title('Amazon PPC Negative Keyword Analyzer')
    st.caption('Upload Amazon Search Term reports and an optional Negative Keyword report to surface reusable single-word and word-pair negative candidates.')

    with st.sidebar:
        st.header('Analysis settings')
        min_clicks_floor = st.number_input('Minimum clicks floor', min_value=1, max_value=100, value=6, help='Minimum clicks needed before a token can be treated as a serious candidate.')
        pair_min_clicks_floor = st.number_input('Word pair minimum clicks floor', min_value=1, max_value=100, value=4)
        low_sales_max = st.number_input('Low sales ceiling', min_value=0.0, max_value=1000.0, value=1.0, step=0.5, help='Treat terms as low-sales when they are below or equal to this amount.')
        low_relevancy_ratio = st.slider('Relative CVR threshold', min_value=0.1, max_value=1.0, value=0.5, step=0.05, help='Flag terms when conversion rate is below this share of campaign baseline CVR.')
        zero_sales_quantile = st.slider('Zero-sales click quantile', min_value=0.5, max_value=0.95, value=0.75, step=0.05)
        low_sales_click_quantile = st.slider('Low-sales click quantile', min_value=0.5, max_value=0.95, value=0.70, step=0.05)
        high_spend_quantile = st.slider('High-spend quantile', min_value=0.5, max_value=0.95, value=0.70, step=0.05)
        exclude_numeric_tokens = st.checkbox('Exclude numeric-only tokens', value=True)

    settings = AnalyzerSettings(
        min_clicks_floor=int(min_clicks_floor),
        low_sales_max=float(low_sales_max),
        low_relevancy_ratio=float(low_relevancy_ratio),
        zero_sales_quantile=float(zero_sales_quantile),
        low_sales_click_quantile=float(low_sales_click_quantile),
        high_spend_quantile=float(high_spend_quantile),
        pair_min_clicks_floor=int(pair_min_clicks_floor),
        exclude_numeric_tokens=bool(exclude_numeric_tokens),
    )

    left, right = st.columns([1.3, 1])
    with left:
        search_term_files = st.file_uploader(
            'Search Term CSV files',
            type=['csv'],
            accept_multiple_files=True,
            help='Upload one or more Amazon Sponsored Products Search Term reports.'
        )
    with right:
        negative_keyword_file = st.file_uploader(
            'Negative Keyword CSV file (optional)',
            type=['csv'],
            help='Upload your current negative keyword export so duplicates can be excluded from recommendations.'
        )

    analyze = st.button('Run analysis', type='primary', use_container_width=True)

    if analyze:
        if not search_term_files:
            st.error('Please upload at least one Search Term CSV file.')
            st.stop()

        try:
            results = analyze_uploaded_files(search_term_files, negative_keyword_file, settings)
        except Exception as exc:
            st.exception(exc)
            st.stop()

        summary = results['summary']
        baseline = summary['campaign_baseline']
        recommendations = results['recommendations']
        single_words = results['single_words']
        word_pairs = results['word_pairs']
        top_terms = results['high_click_zero_sale_terms']

        st.success('Analysis complete.')

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('Unique search terms', f"{summary['rows_search_terms']:,}")
        m2.metric('Negative keywords loaded', f"{summary['rows_negative_keywords']:,}")
        m3.metric('Recommended candidates', f"{summary['recommended_candidates']:,}")
        m4.metric('Campaign CVR', f"{baseline['cvr']:.2%}")
        m5.metric('Avg CPC', f"${baseline['avg_cpc']:.2f}")

        st.subheader('Top recommended negatives')
        if recommendations.empty:
            st.info('No candidates met the current thresholds. Try lowering the click floors or the relative CVR threshold.')
        else:
            preview_cols = ['token', 'token_type', 'clicks', 'spend', 'sales', 'orders', 'relative_cvr', 'score', 'recommendation', 'reason']
            st.dataframe(
                recommendations[preview_cols].head(200),
                use_container_width=True,
                hide_index=True,
            )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            'Recommended candidates',
            'Single words',
            'Word pairs',
            'High-click zero-sale terms',
            'Summary JSON',
        ])

        with tab1:
            st.dataframe(recommendations, use_container_width=True, hide_index=True)
            st.download_button(
                'Download recommended_negative_candidates.csv',
                data=recommendations.to_csv(index=False).encode('utf-8'),
                file_name='recommended_negative_candidates.csv',
                mime='text/csv'
            )

        with tab2:
            st.dataframe(single_words, use_container_width=True, hide_index=True)
            st.download_button(
                'Download single_word_analysis.csv',
                data=single_words.to_csv(index=False).encode('utf-8'),
                file_name='single_word_analysis.csv',
                mime='text/csv'
            )

        with tab3:
            st.dataframe(word_pairs, use_container_width=True, hide_index=True)
            st.download_button(
                'Download word_pair_analysis.csv',
                data=word_pairs.to_csv(index=False).encode('utf-8'),
                file_name='word_pair_analysis.csv',
                mime='text/csv'
            )

        with tab4:
            st.dataframe(top_terms, use_container_width=True, hide_index=True)
            st.download_button(
                'Download high_click_zero_sale_search_terms.csv',
                data=top_terms.to_csv(index=False).encode('utf-8'),
                file_name='high_click_zero_sale_search_terms.csv',
                mime='text/csv'
            )

        with tab5:
            st.code(json.dumps(summary, indent=2), language='json')

        zip_bytes = build_download_zip(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            'Download full output ZIP',
            data=zip_bytes,
            file_name=f'amazon_ppc_analysis_output_{timestamp}.zip',
            mime='application/zip',
            use_container_width=True,
        )

        st.markdown('### How to use the output')
        st.markdown(
            '- Start with **Recommended candidates** for the shortest review list.\n'
            '- Use **Single words** and **Word pairs** to spot repeated intent patterns.\n'
            '- Use **High-click zero-sale terms** as a final manual review layer before adding negatives in Amazon Ads.'
        )
    else:
        st.info('Upload your files, adjust thresholds if needed, and click **Run analysis**.')
        st.markdown(
            'This app stays campaign-agnostic by detecting columns dynamically and scoring terms relative to each uploaded campaign dataset rather than relying on campaign-specific rules.'
        )


if __name__ == '__main__':
    main()
