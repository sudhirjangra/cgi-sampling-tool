import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Sampling Tool [@sudhirjangra]", 
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    .success-box {padding:10px; background-color:#e6fffa; border-left:5px solid #00cc99; border-radius:5px;}
    .warning-box {padding:10px; background-color:#fff4e6; border-left:5px solid #ff9933; border-radius:5px;}
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions (Cached for Performance) ---

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Loads Excel file efficiently."""
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None

@st.cache_data(show_spinner=False)
def process_data_logic(df, value_cols, mode):
    """Performs metrics calculation based on strict criteria."""
    df_out = df.copy()
    
    # ---------------------------------------------------------
    # CRITERIA 1: Downtime Sampling Logic
    # ---------------------------------------------------------
    if mode == "Downtime Sampling":
        # 1. Fill Blanks/NaNs with 0
        df_out[value_cols] = df_out[value_cols].fillna(0)

        # 2. Calculate Count (Number of days > 0)
        temp_nums = df_out[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df_out["Result"] = (temp_nums > 0).sum(axis=1)
        
        # 3. Binning (0-10, 10-20, 20-30, 30+)
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

    # ---------------------------------------------------------
    # CRITERIA 2: CGI (DCR) Sampling Logic
    # ---------------------------------------------------------
    else: 
        # 1. Calculate Row-wise Average
        # Note: We prioritize the numeric average. 
        # If numbers exist, we take the mean. Strings are ignored in the mean.
        # If ONLY strings exist (e.g. 'DNE' everywhere), result becomes NaN initially.
        df_numeric = df_out[value_cols].apply(pd.to_numeric, errors='coerce')
        df_out["Result"] = df_numeric.mean(axis=1, skipna=True)
        
        # 2. Handle Non-Numeric Rows (Strings like "A", "DNE")
        # If Result is NaN (no numbers found), grab the first string available.
        def fill_string_fallback(row):
            if pd.isna(row["Result"]):
                # Look for strings in original data
                vals = row[value_cols]
                strs = [v for v in vals if isinstance(v, str) and str(v).strip() != '']
                return strs[0] if strs else None
            return row["Result"]

        if df_out["Result"].isna().any():
            df_out["Result"] = df_out.apply(fill_string_fallback, axis=1)
        
        # 3. Binning (Best/Variable/Worst)
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
        
        # 4. Handle Strings in Bins (e.g. "DNE" maps to "DNE" bin/label)
        df_out["Result_bin"] = df_out["Result_bin"].astype(object)
        mask = df_out["Result_bin"].isna()
        df_out.loc[mask, "Result_bin"] = df_out.loc[mask, "Result"]

    # Final safeguard
    df_out["Result_bin"] = df_out["Result_bin"].astype(object)
    return df_out

@st.cache_data(show_spinner=False)
def perform_sampling(df, strat_cols, target_size, col_tech, min_tech_req):
    """
    Constraint-First Sampling with Proportional Fill:
    1. Coverage: Get 1 sample from every unique Stratification Group.
    2. Tech Quota: Ensure every Technology has at least `min_tech_req`.
    3. Target Fill: Fill remaining slots PROPORTIONALLY to preserve distribution.
    """
    target_size = int(target_size)
    min_tech_req = int(min_tech_req)
    
    # 1. Stratified Sample (Priority 1: Coverage)
    try:
        base_sample = df.groupby(strat_cols, group_keys=False).sample(n=1, random_state=24)
    except Exception:
        base_sample = df.groupby(strat_cols, group_keys=False).apply(lambda x: x.sample(1, random_state=24))
    
    current_sample = base_sample.copy()
    guaranteed_count = len(current_sample)

    # 2. Tech Quota (Priority 2: Minimum Requirement)
    if col_tech and min_tech_req > 0:
        boost_rows = []
        tech_counts = current_sample[col_tech].value_counts()
        unique_techs = df[col_tech].unique()
        
        for tech in unique_techs:
            current_count = tech_counts.get(tech, 0)
            if current_count < min_tech_req:
                needed = min_tech_req - current_count
                
                # Find available rows not yet in sample
                mask = (~df.index.isin(current_sample.index)) & (df[col_tech] == tech)
                available = df[mask]
                
                if not available.empty:
                    take = min(needed, len(available))
                    boost = available.sample(n=take, random_state=42)
                    boost_rows.append(boost)
        
        if boost_rows:
            all_boost = pd.concat(boost_rows)
            current_sample = pd.concat([current_sample, all_boost])

    # 3. Target Fill (Priority 3: Reach Target Size with Proportional Sampling)
    current_len = len(current_sample)
    remaining = target_size - current_len
    
    if remaining > 0:
        mask = ~df.index.isin(current_sample.index)
        pool = df[mask]
        
        if not pool.empty:
            if len(pool) <= remaining:
                # Take everything if pool is smaller than need
                fill = pool
            else:
                # Use strat_cols for weights to maintain distribution?
                # A simple random sample from the remaining pool generally preserves distribution
                # unless the 'boost' step heavily skewed it.
                # To be safe, we sample randomly.
                fill = pool.sample(n=remaining, random_state=52)
                
            current_sample = pd.concat([current_sample, fill])
            
    # 4. Final Shuffle
    final_df = current_sample.sample(frac=1, random_state=99).reset_index(drop=True)
    return final_df, guaranteed_count

def convert_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sampled Data')
    return output.getvalue()

def create_comparison_table(df_pop, df_sample, col_name, top_n=None):
    """Creates a side-by-side comparison of counts."""
    # Population Stats
    pop_counts = df_pop[col_name].value_counts().reset_index()
    pop_counts.columns = [col_name, "Pop Count"]
    pop_counts["Pop %"] = (pop_counts["Pop Count"] / len(df_pop) * 100).round(1)
    
    # Sample Stats
    samp_counts = df_sample[col_name].value_counts().reset_index()
    samp_counts.columns = [col_name, "Sample Count"]
    samp_counts["Sample %"] = (samp_counts["Sample Count"] / len(df_sample) * 100).round(1)
    
    # Merge
    merged = pd.merge(pop_counts, samp_counts, on=col_name, how="outer").fillna(0)
    
    # Sort
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

# --- Main UI ---

def main():
    st.sidebar.title("⚙️ Configuration")
    
    # 1. Mode Selection
    mode = st.sidebar.selectbox("Sampling Mode", ["CGI Sampling", "Downtime Sampling"], 
                                help="CGI: Average/Quality Metric\nDowntime: Zero-fill count metric")
    
    # 2. File Upload
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

    # 3. Settings
    with st.sidebar.expander("📝 Data Mapping", expanded=True):
        if uploaded_file:
            df = load_data(uploaded_file)
            cols = df.columns.tolist()
            
            # Auto-mapping
            def get_idx(search_terms, default_idx):
                for i, c in enumerate(cols):
                    if any(s in c.lower() for s in search_terms): return i
                return default_idx if default_idx < len(cols) else 0

            p_idx = get_idx(['pincode', 'pin'], 1)
            d_idx = get_idx(['district', 'dist'], 2)
            t_idx = get_idx(['tech', 'rat'], 5)
            
            c_pin = st.selectbox("Pincode", cols, index=p_idx)
            c_dist = st.selectbox("District", cols, index=d_idx)
            c_tech = st.selectbox("Technology", cols, index=t_idx)
            
            # Technology Filter
            if c_tech:
                unique_techs = df[c_tech].unique().tolist()
                st.caption("🔍 Filter Technologies")
                selected_techs = st.multiselect(
                    "Include Technologies", 
                    options=unique_techs, 
                    default=unique_techs,
                    help="Uncheck values like '0', 'nan' to exclude them."
                )
            
            st.divider()
            
            # Date Range Selection
            st.write("**Select Date Range**")
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
                st.caption(f"Selected {len(val_cols)} columns (from '{c_start}' to '{c_end}').")

        else:
            st.info("Upload file to see settings.")
            df = None
            selected_techs = []

    with st.sidebar.expander("🎯 Targets", expanded=True):
        default_t = 2000 if mode == "CGI Sampling" else 1500
        target_size = st.number_input("Target Sample Size", 100, 100000, default_t, step=100)
        min_tech = st.number_input("Min Rows per Tech", 0, 5000, 385, help="Constraint: Must have at least this many rows per technology.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Developed by Sudhir Jangra")

    # --- Main Content ---
    st.title("⚡ Advanced Sampling Tool")

    if df is not None and val_cols:
        if st.sidebar.button("🚀 Run Sampling Analysis", type="primary"):
            
            with st.status("Processing data...", expanded=True) as status:
                # 0. Filter Technologies
                if len(selected_techs) < len(unique_techs):
                    df_filtered = df[df[c_tech].isin(selected_techs)].copy()
                else:
                    df_filtered = df.copy()

                st.write("1. Calculating metrics (Row-wise Average)...")
                df_proc = process_data_logic(df_filtered, val_cols, mode)
                
                st.write("2. Performing Constraint-Based Sampling...")
                strat_cols = [c_pin, c_dist, c_tech, "Result_bin"]
                sampled, guaranteed = perform_sampling(df_proc, strat_cols, target_size, c_tech, min_tech)
                
                if "num_Result" in sampled.columns:
                    sampled = sampled.drop(columns=["num_Result"])
                    
                status.update(label="Sampling Complete!", state="complete", expanded=False)

            # --- Metrics Row ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Sample Size", f"{len(sampled):,}", f"{len(sampled)/len(df):.1%} of total")
            col2.metric("Guaranteed Coverage", f"{guaranteed:,}", help="Rows ensuring 100% Stratification coverage")
            col3.metric("Original Size", f"{len(df):,}")
            col4.metric("Mode", mode.split()[0])
            
            if len(sampled) > target_size:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <b>Note:</b> Final sample size ({len(sampled)}) exceeded your target ({target_size}).
                    <br>This happened because the <b>Minimum Tech Constraint ({min_tech})</b> 
                    and <b>Coverage Requirements</b> required more data.
                </div>
                """, unsafe_allow_html=True)

            # --- Tabs ---
            tab_cov, tab_dist, tab_data = st.tabs(["🌍 Coverage Analysis", "📊 Distribution Charts", "💾 Sampled Data"])

            # --- TAB 1: Coverage Analysis (Unique Counts) ---
            with tab_cov:
                st.subheader("Coverage: Population vs. Sample")
                
                # Metrics for Unique Counts
                c1, c2, c3 = st.columns(3)
                c1.metric("Unique Pincodes", f"{sampled[c_pin].nunique()}", f"of {df_proc[c_pin].nunique()} Total")
                c2.metric("Unique Districts", f"{sampled[c_dist].nunique()}", f"of {df_proc[c_dist].nunique()} Total")
                c3.metric("Unique Technologies", f"{sampled[c_tech].nunique()}", f"of {df_proc[c_tech].nunique()} Total")
                
                st.divider()
                
                # Side-by-Side Charts
                c1, c2 = st.columns(2)
                
                with c1:
                    st.write("**Top 15 Districts Covered (Volume)**")
                    dist_comp = create_comparison_table(df_proc, sampled, c_dist, top_n=15)
                    dist_long = pd.melt(dist_comp, id_vars=[c_dist], value_vars=["Pop Count", "Sample Count"], 
                                      var_name="Dataset", value_name="Count")
                    fig_d = px.bar(dist_long, x=c_dist, y="Count", color="Dataset", barmode="group",
                                   color_discrete_map={"Pop Count": "#d3d3d3", "Sample Count": "#FF9800"})
                    st.plotly_chart(fig_d, use_container_width=True)
                    
                with c2:
                    st.write("**Top 15 Pincodes Covered (Volume)**")
                    pin_comp = create_comparison_table(df_proc, sampled, c_pin, top_n=15)
                    pin_long = pd.melt(pin_comp, id_vars=[c_pin], value_vars=["Pop Count", "Sample Count"], 
                                      var_name="Dataset", value_name="Count")
                    pin_long[c_pin] = pin_long[c_pin].astype(str)
                    fig_p = px.bar(pin_long, x=c_pin, y="Count", color="Dataset", barmode="group",
                                   color_discrete_map={"Pop Count": "#d3d3d3", "Sample Count": "#9C27B0"})
                    st.plotly_chart(fig_p, use_container_width=True)

            # --- TAB 2: Distribution Charts (Percentage) ---
            with tab_dist:
                st.subheader("Distribution Comparison (%)")
                c1, c2 = st.columns(2)
                
                def prep_chart_data(df_comp, col_name):
                    return pd.melt(df_comp, id_vars=[col_name], value_vars=["Pop %", "Sample %"], 
                                   var_name="Dataset", value_name="Percentage")

                with c1:
                    st.write("**Result Bin Distribution**")
                    df_res_comp = create_comparison_table(df_proc, sampled, "Result_bin")
                    chart_data = prep_chart_data(df_res_comp, "Result_bin")
                    
                    if mode == "Downtime Sampling":
                        order = ["0–10", "10–20", "20–30", "30+"]
                    else:
                        order = ["Best DCR", "Variable DCR", "Worst DCR"]

                    fig = px.bar(chart_data, x="Result_bin", y="Percentage", color="Dataset", barmode="group",
                                 category_orders={"Result_bin": order},
                                 color_discrete_map={"Pop %": "#d3d3d3", "Sample %": "#4CAF50"})
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    st.write("**Technology Mix**")
                    df_tech_comp = create_comparison_table(df_proc, sampled, c_tech)
                    chart_data_tech = prep_chart_data(df_tech_comp, c_tech)
                    fig2 = px.bar(chart_data_tech, x=c_tech, y="Percentage", color="Dataset", barmode="group",
                                  color_discrete_map={"Pop %": "#d3d3d3", "Sample %": "#2196F3"})
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    min_check = df_tech_comp["Sample Count"].min()
                    if min_check >= min_tech:
                        st.success(f"✅ All technologies met the {min_tech} minimum requirement.")
                    else:
                        st.warning(f"⚠️ Some technologies have < {min_tech} rows (Source data insufficient).")

            # --- TAB 3: Data Table ---
            with tab_data:
                st.dataframe(sampled, use_container_width=True)
                st.info(f"✅ Calculated over columns: {c_start} to {c_end}")

            # --- Download ---
            st.markdown("### 📥 Export")
            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                excel_data = convert_to_excel(sampled)
                fname = f"Sampled_{mode.split()[0]}_{uploaded_file.name}"
                st.download_button("Download Excel Result", excel_data, fname, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    else:
        st.markdown("""
        <div style='text-align: center; color: #666; margin-top: 50px;'>
            <h3>👋 Welcome to the Sampling Tool</h3>
            <p>Upload a file in the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()