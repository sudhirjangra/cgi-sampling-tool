import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os

st.set_page_config(
    page_title="Advanced Sampling Tool", 
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    .success-box {padding:10px; background-color:#e6fffa; border-left:5px solid #00cc99; border-radius:5px;}
    .warning-box {padding:10px; background-color:#fff4e6; border-left:5px solid #ff9933; border-radius:5px;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_sheet_names(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            xl = pd.ExcelFile(uploaded_file)
            return xl.sheet_names
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_data(uploaded_file, sheet_name=None):
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                return pd.read_csv(uploaded_file, encoding='latin1')
        else:
            return pd.read_excel(uploaded_file, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None

@st.cache_data(show_spinner=False)
def process_data_logic(df, value_cols, mode):
    df_out = df.copy()
    if mode == "Downtime Sampling":
        df_out[value_cols] = df_out[value_cols].fillna(0)
        temp_nums = df_out[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df_out["Result"] = (temp_nums > 0).sum(axis=1)
        bins = [0, 10, 20, 30, float("inf")]
        labels = ["0–10", "10–20", "20–30", "30+"]
        df_out["Result_bin"] = pd.cut(
            df_out["Result"], 
            bins=bins, 
            labels=labels, 
            include_lowest=True, 
            right=False 
        )
        df_out["Result_bin"] = df_out["Result_bin"].astype(object).fillna("0–10")
    else: 
        df_numeric = df_out[value_cols].apply(pd.to_numeric, errors='coerce')
        df_out["Result"] = df_numeric.mean(axis=1, skipna=True)
        def fill_string_fallback(row):
            if pd.isna(row["Result"]):
                vals = row[value_cols]
                strs = [v for v in vals if isinstance(v, str) and str(v).strip() != '']
                return strs[0] if strs else None
            return row["Result"]
        if df_out["Result"].isna().any():
            df_out["Result"] = df_out.apply(fill_string_fallback, axis=1)
        df_out["num_Result"] = pd.to_numeric(df_out["Result"], errors="coerce")
        bins = [0, 0.1, 1, float("inf")]
        labels = ["Best DCR", "Variable DCR", "Worst DCR"]
        df_out["Result_bin"] = pd.cut(
            df_out["num_Result"], 
            bins=bins, 
            labels=labels, 
            include_lowest=True, 
            right=True
        )
        df_out["Result_bin"] = df_out["Result_bin"].astype(object)
        mask = df_out["Result_bin"].isna()
        df_out.loc[mask, "Result_bin"] = df_out.loc[mask, "Result"]
    df_out["Result_bin"] = df_out["Result_bin"].astype(object)
    return df_out

@st.cache_data(show_spinner=False)
def perform_sampling(df, strat_cols, target_size, col_tech, min_tech_req):
    target_size = int(target_size)
    min_tech_req = int(min_tech_req) if col_tech else 0
    valid_strat_cols = [c for c in strat_cols if c is not None and c in df.columns]
    try:
        if not valid_strat_cols:
             base_sample = df.sample(n=min(1, len(df)), random_state=24)
        else:
            base_sample = df.groupby(valid_strat_cols, group_keys=False).sample(n=1, random_state=24)
    except Exception:
        base_sample = df.groupby(valid_strat_cols, group_keys=False).apply(lambda x: x.sample(1, random_state=24))
    current_sample = base_sample.copy()
    guaranteed_count = len(current_sample)
    if col_tech and min_tech_req > 0 and col_tech in df.columns:
        boost_rows = []
        tech_counts = current_sample[col_tech].value_counts()
        unique_techs = df[col_tech].unique()
        for tech in unique_techs:
            current_count = tech_counts.get(tech, 0)
            if current_count < min_tech_req:
                needed = min_tech_req - current_count
                mask = (~df.index.isin(current_sample.index)) & (df[col_tech] == tech)
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
                fill = pool.sample(n=remaining, random_state=52)
            current_sample = pd.concat([current_sample, fill])
    final_df = current_sample.sample(frac=1, random_state=99).reset_index(drop=True)
    return final_df, guaranteed_count

def convert_to_excel(df):
    output = io.BytesIO()
    df_export = df.copy()
    for col in df_export.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns:
        if hasattr(df_export[col], 'dt') and df_export[col].dt.tz is not None:
             df_export[col] = df_export[col].dt.tz_localize(None)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Sampled Data')
    return output.getvalue()

def create_comparison_table(df_pop, df_sample, col_name, top_n=None):
    if col_name not in df_pop.columns or col_name is None:
        return pd.DataFrame()
    pop_counts = df_pop[col_name].value_counts().reset_index()
    pop_counts.columns = [col_name, "Pop Count"]
    pop_counts["Pop %"] = (pop_counts["Pop Count"] / len(df_pop) * 100).round(1)
    samp_counts = df_sample[col_name].value_counts().reset_index()
    samp_counts.columns = [col_name, "Sample Count"]
    samp_counts["Sample %"] = (samp_counts["Sample Count"] / len(df_sample) * 100).round(1)
    merged = pd.merge(pop_counts, samp_counts, on=col_name, how="outer").fillna(0)
    if top_n:
        merged = merged.sort_values("Pop Count", ascending=False).head(top_n)
    elif "0–10" in str(merged[col_name].iloc[0]): 
        order = ["0–10", "10–20", "20–30", "30+"]
        merged[col_name] = pd.Categorical(merged[col_name], categories=order, ordered=True)
        merged = merged.sort_values(col_name)
    elif "Best DCR" in str(merged[col_name].iloc[0]):
        order = ["Best DCR", "Variable DCR", "Worst DCR"]
        merged[col_name] = pd.Categorical(merged[col_name], categories=order, ordered=True)
        merged = merged.sort_values(col_name)
    else:
        merged = merged.sort_values("Pop Count", ascending=False)
    return merged

def main():
    st.sidebar.title("⚙️ Configuration")
    mode = st.sidebar.selectbox("Sampling Mode", ["CGI Sampling", "Downtime Sampling"])
    uploaded_file = st.sidebar.file_uploader("Upload Data File", type=["xlsx", "xls", "csv"])

    selected_sheet = None
    if uploaded_file:
        orig_filename = os.path.splitext(uploaded_file.name)[0]
        sheets = get_sheet_names(uploaded_file)
        if sheets and len(sheets) > 1:
            selected_sheet = st.sidebar.selectbox("Select Worksheet", sheets)
        else:
            selected_sheet = sheets[0] if sheets else None
        df = load_data(uploaded_file, selected_sheet)
    else:
        df = None
        orig_filename = "Result"

    with st.sidebar.expander("📝 Data Mapping", expanded=True):
        if uploaded_file and df is not None:
            cols = df.columns.tolist()
            def get_idx(search_terms, default_idx):
                for i, c in enumerate(cols):
                    if any(s in c.lower() for s in search_terms): return i
                return None 
            d_idx = get_idx(['district', 'dist'], 2)
            p_idx = get_idx(['pincode', 'pin'], 1)
            t_idx = get_idx(['tech', 'rat'], 5)
            d_default = d_idx if d_idx is not None else 0
            c_dist = st.selectbox("District (Mandatory)", cols, index=d_default)
            p_default = p_idx if p_idx is not None else -1 
            pin_options = ["None"] + cols
            pin_index = (cols.index(cols[p_default]) + 1) if p_default != -1 else 0
            c_pin_sel = st.selectbox("Pincode (Optional)", pin_options, index=pin_index)
            c_pin = None if c_pin_sel == "None" else c_pin_sel
            t_default = t_idx if t_idx is not None else -1
            tech_options = ["None"] + cols
            tech_index = (cols.index(cols[t_default]) + 1) if t_default != -1 else 0
            c_tech_sel = st.selectbox("Technology (Optional)", tech_options, index=tech_index)
            c_tech = None if c_tech_sel == "None" else c_tech_sel
            selected_techs = []
            if c_tech:
                unique_techs = df[c_tech].unique().tolist()
                st.caption("🔍 Filter Technologies")
                selected_techs = st.multiselect("Include Technologies", options=unique_techs, default=unique_techs)
            st.divider()
            st.write("**Select Date Range (Mandatory)**")
            def_start = 8 if len(cols) > 8 else 0
            col_s, col_e = st.columns(2)
            with col_s:
                c_start = st.selectbox("Start Date Column", cols, index=def_start)
            with col_e:
                def_end = len(cols) - 1
                c_end = st.selectbox("End Date Column", cols, index=def_end)
            idx_start = cols.index(c_start)
            idx_end = cols.index(c_end)
            if idx_end < idx_start:
                st.error("End column must be after Start column!")
                val_cols = []
            else:
                val_cols = cols[idx_start : idx_end + 1]
                st.caption(f"Selected {len(val_cols)} columns.")
        else:
            st.info("Upload file to see settings.")
            val_cols = []
            selected_techs = []
            c_tech = None

    with st.sidebar.expander("🎯 Targets", expanded=True):
        target_type = st.radio("Target Logic", ["Fixed Number", "Percentage (%)"], horizontal=True)
        target_size = 0
        pct = 20
        if target_type == "Percentage (%)":
            pct = st.slider("Percentage of Population", 1, 100, 20)
            if df is not None:
                approx = int(len(df) * (pct / 100))
                st.caption(f"Approx Target (Pre-filter): ~{approx} rows")
            else:
                target_size = 100
        else:
            default_t = 2000 if mode == "CGI Sampling" else 1500
            target_size = st.number_input("Target Sample Size", 100, 1000000, default_t, step=100)
        min_tech = 0
        if c_tech:
            min_tech = st.number_input("Min Rows per Tech", 0, 5000, 385)
        else:
            st.info("Min Tech constraint disabled.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Developed by Sudhir Jangra")
    st.title("⚡ Advanced Sampling Tool")

    if df is not None and val_cols:
        if st.sidebar.button("🚀 Run Sampling Analysis", type="primary"):
            with st.status("Processing data...", expanded=True) as status:
                if c_tech and len(selected_techs) > 0:
                    df_filtered = df[df[c_tech].isin(selected_techs)].copy()
                else:
                    df_filtered = df.copy()
                if target_type == "Percentage (%)":
                    target_size = max(1, int(len(df_filtered) * (pct / 100)))
                df_proc = process_data_logic(df_filtered, val_cols, mode)
                strat_cols = [c_dist, "Result_bin"]
                if c_pin:
                    strat_cols.insert(0, c_pin)
                if c_tech:
                    strat_cols.append(c_tech)
                sampled, guaranteed = perform_sampling(df_proc, strat_cols, target_size, c_tech, min_tech)
                if "num_Result" in sampled.columns:
                    sampled = sampled.drop(columns=["num_Result"])
                status.update(label="Sampling Complete!", state="complete", expanded=False)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Sample Size", f"{len(sampled):,}", f"{len(sampled)/len(df_filtered):.1%} of filtered")
            col2.metric("Guaranteed Coverage", f"{guaranteed:,}")
            col3.metric("Population Size", f"{len(df_filtered):,}")
            col4.metric("Mode", mode.split()[0])
            
            if len(sampled) > target_size * 1.05:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <b>Note:</b> Final sample size ({len(sampled)}) exceeded your target ({target_size}).
                </div>
                """, unsafe_allow_html=True)

            tab_cov, tab_dist, tab_data = st.tabs(["🌍 Coverage Analysis", "📊 Distribution Charts", "💾 Sampled Data"])

            with tab_cov:
                st.subheader("Coverage: Population vs. Sample")
                c1, c2, c3 = st.columns(3)
                if c_pin:
                    c1.metric("Unique Pincodes", f"{sampled[c_pin].nunique()}", f"of {df_proc[c_pin].nunique()} Total")
                else:
                    c1.metric("Unique Pincodes", "N/A")
                c2.metric("Unique Districts", f"{sampled[c_dist].nunique()}", f"of {df_proc[c_dist].nunique()} Total")
                if c_tech:
                    c3.metric("Unique Technologies", f"{sampled[c_tech].nunique()}", f"of {df_proc[c_tech].nunique()} Total")
                else:
                    c3.metric("Unique Technologies", "N/A")
                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Top 15 Districts Covered (Volume)**")
                    dist_comp = create_comparison_table(df_proc, sampled, c_dist, top_n=15)
                    if not dist_comp.empty:
                        dist_long = pd.melt(dist_comp, id_vars=[c_dist], value_vars=["Pop Count", "Sample Count"], var_name="Dataset", value_name="Count")
                        fig_d = px.bar(dist_long, x=c_dist, y="Count", color="Dataset", barmode="group", color_discrete_map={"Pop Count": "#d3d3d3", "Sample Count": "#FF9800"})
                        st.plotly_chart(fig_d, use_container_width=True)
                with c2:
                    if c_pin:
                        st.write("**Top 15 Pincodes Covered (Volume)**")
                        pin_comp = create_comparison_table(df_proc, sampled, c_pin, top_n=15)
                        if not pin_comp.empty:
                            pin_long = pd.melt(pin_comp, id_vars=[c_pin], value_vars=["Pop Count", "Sample Count"], var_name="Dataset", value_name="Count")
                            pin_long[c_pin] = pin_long[c_pin].astype(str)
                            fig_p = px.bar(pin_long, x=c_pin, y="Count", color="Dataset", barmode="group", color_discrete_map={"Pop Count": "#d3d3d3", "Sample Count": "#9C27B0"})
                            st.plotly_chart(fig_p, use_container_width=True)

            with tab_dist:
                st.subheader("Distribution Comparison (%)")
                c1, c2 = st.columns(2)
                def prep_chart_data(df_comp, col_name):
                    return pd.melt(df_comp, id_vars=[col_name], value_vars=["Pop %", "Sample %"], var_name="Dataset", value_name="Percentage")
                with c1:
                    st.write("**Result Bin Distribution**")
                    df_res_comp = create_comparison_table(df_proc, sampled, "Result_bin")
                    chart_data = prep_chart_data(df_res_comp, "Result_bin")
                    order = ["0–10", "10–20", "20–30", "30+"] if mode == "Downtime Sampling" else ["Best DCR", "Variable DCR", "Worst DCR"]
                    fig = px.bar(chart_data, x="Result_bin", y="Percentage", color="Dataset", barmode="group", category_orders={"Result_bin": order}, color_discrete_map={"Pop %": "#d3d3d3", "Sample %": "#4CAF50"})
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if c_tech:
                        st.write("**Technology Mix**")
                        df_tech_comp = create_comparison_table(df_proc, sampled, c_tech)
                        if not df_tech_comp.empty:
                            chart_data_tech = prep_chart_data(df_tech_comp, c_tech)
                            fig2 = px.bar(chart_data_tech, x=c_tech, y="Percentage", color="Dataset", barmode="group", color_discrete_map={"Pop %": "#d3d3d3", "Sample %": "#2196F3"})
                            st.plotly_chart(fig2, use_container_width=True)

            with tab_data:
                st.dataframe(sampled, use_container_width=True)

            st.markdown("### 📥 Export")
            excel_data = convert_to_excel(sampled)
            export_name = selected_sheet if selected_sheet else orig_filename
            st.download_button("Download Excel Result", excel_data, f"Sampled_{export_name}.xlsx", type="primary")
    else:
        st.markdown("<div style='text-align: center; color: #666; margin-top: 50px;'><h3>👋 Welcome to the Sampling Tool</h3><p>Upload a file to get started.</p></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()