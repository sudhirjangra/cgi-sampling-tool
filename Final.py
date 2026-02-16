import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="Sampling Tool [@sudhirjangra]",
    layout='wide',
    page_icon="⚡",
    initial_sidebar_state='auto',  
)

st.markdown("""
            <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    .success-box {padding:10px; background-color:#e6fffa; border-left:5px solid #00cc99; border-radius:5px;}
    .warning-box {padding:10px; background-color:#fff4e6; border-left:5px solid #ff9933; border-radius:5px;}
    </style>
            """, unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def load_data(uploaded_file):
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None
    
@st.cache_data(show_spinner=True)
def process_data_login(df, value_cols, mode):
    df_out = df.copy()
    if mode == "Downtime Sampling":
        df_out[value_cols] = df_out[value_cols].fillna(0)
        temp_nums = df_out[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df_out["Result"] = (temp_nums > 0).sum(axis=1)
        bins = [0,10,20,30,float("inf")]
        labels = ['0-10', '10-20','20-30','30+']
        
        df_out['Result_bin'] = pd.cut(
            df_out['Result'],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False
        )
        
        df_out['Result_bin'] = df.out['Result_bin'].astype(object).fillna("0-10")
    
    else:
        df_numeric = df_out[value_cols].apply(pd.to_numeric, errors='coerce')
        df_out['Result'] = df_numeric.mean(axis=1, skipna=True)
        
        def fill_string_fallback(row):
            if pd.isna(row['Result']):
                vals = row[value_cols]
                strs = [v for v in vals if isinstance(v,str) and str(v).strip()!='']
                return strs[0] if strs else None
            return row['Result']
        
        if df_out['Result'].isna().any():
            df_out['Result'] = df_out.apply(fill_string_fallback, axis=1)
            
        df_out['num_Result'] = pd.to_numeric(df_out['Result'], errors='coerce')
        bins = [0,0.1,1,float('inf')]
        labels = ['Best DCR', 'Variable DCR', 'Worst DCR']
        
        df_out["Result_bin"] = pd.cut(
            df_out['num_Result'],
            bins=bins,
            labels = labels,
            include_lowest=True,
            right=True
        )
        
        df_out['Result_bin'] = df_out['Result_bin'].astype(object)
        mask = df_out['Result_bin'].isna()
        df_out.loc[mask, 'Result_bin'] = df_out.loc[mask, 'Result']
        
    df_out['Result_bin'] = df_out['Result_bin'].astype(object)
    return df_out

@st.cache_data(show_spinner=True)
def perform_sampling(df, strat_cols, target_size, col_tech, min_tech_req):
    """
    Constraint-First Sampling with Proportional Fill:
    1. Coverage: Get 1 sample from every unique Stratification Group.
    2. Tech Quota: Ensure every Technology has at least `min_tech_req`.
    3. Target Fill: Fill remaining slots PROPORTIONALLY to preserve distribution.
    """
    target_size = int(target_size)
    min_tech_req = int(min_tech_req)
    
    try:
        base_sample = df.groupby(strat_cols, group_keys = False)
    except Exception:
        base_sample = df.groupby(strat_cols, group_keys = False)
        
    current_sample = base_sample.copy()
    guaranteed_count = len(current_sample)
    
    if col_tech and min_tech_req > 0:
        boost_rows = []
        tech_counts = current_sample[col_tech].value_counts()
        unique_techs = df[col_tech].unique()
        
        for tech in unique_techs:
            current_count = tech_counts.get(tech, 0)
            if current_count < min_tech_req:
                needed = min_tech_req - current_count
                
                mask = (~df.index.isin(current_count.index))
                available = df[mask]
                
                if not available.empty:
                    take = min(needed, len(available))
                    boost = available.sample(n=take, random_state=42)
                    boost_rows.append(boost)
                    
        if boost_rows:
            all_boost = pd.concat(boost_rows)
            current_sample = pd.concat([current_sample, all_boost])
    
    current_len = len(current_sample)
    remaining = target_size - current_len
    
    if remaining > 0:
        mask = ~df.index.isin(current_sample.index)
        pool = df[mask]
        
        if not pool.empty:
            if len(pool) <= remaining:
                fill = pool
            else:
                fill = pool.sample(n=remaining, random_state = 52)
            current_sample = pd.concat([current_sample, fill])
            
    
    final_df = current_sample.sample(frac=1, random_state=99).reset_index(drop=True)
    return final_df, guaranteed_count

def convert_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index = False, sheet_name = 'Sampled Data')
    return output.getvalue()

def create_comparison_table(df_pop, df_sample, col_name, top_n=None):
    """Creates a side-by-side comparison of counts."""
    
    pop_counts = df_sample[col_name].value_counts().reset_insex()
    pop_counts.columns = [col_name, "Pop Count"]
    pop_counts["Pop %"] = (pop_counts['Pop Count'] / len(df_pop)*100).round(1)
    
    samp_counts = df_sample[col_name].value_counts().reset_index()
    samp_counts.columns = [col_name, "Sample Count"]
    samp_counts["Sample %"] = (samp_counts['Sample Count']/len(df_sample) * 100).round(1)
    
    merged = pd.merge(pop_counts, samp_counts, on=col_name, how='outer').fillna(0)
    
    if ton_n:
        merged: merged.sort_values("Pop Count", ascending=False).head(top_n)
    elif "0-10" in str(merged[col_name].iloc[0]):
        order = ['0-10','10-20','20-30','30+']
        merged[col_name] = pd.Categorical(merged[col_name], categories=order, ordered=True)
        merged = merged.sort_values(col_name)
    elif "Best DCR" in str(merged[col_name].iloc[0]):
        order = ["Best DCR", "Variable DCR", "Worst DCR"]
        merged[col_name] = pd.Categorical(merged(col_name), categories=order, ordered=True)
        merged = merged.sort_values(col_name)
    else:
        merged = merged.sort_values("Pop Count", ascending=False)
    return merged