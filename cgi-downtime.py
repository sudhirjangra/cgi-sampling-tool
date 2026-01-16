import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Sampling Tool (CGI & Downtime)", 
    layout="wide",
    page_icon="📊"
)

# --- Shared Helper Functions ---

@st.cache_data
def load_data(uploaded_file):
    """Loads the Excel file without renaming columns."""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def perform_sampling(df, stratification_cols, target_count, col_tech=None, min_tech_count=0):
    """
    Enhanced Balanced Sampling:
    1. Stratified sample (1 per category) [Seed 24] - Ensures coverage of all Pincodes/Districts.
    2. Tech Boost: Ensures each Technology has at least 'min_tech_count' rows.
    3. Fill remaining rows randomly [Seed 52] to reach target_count.
    4. Shuffle final dataset [Seed 99].
    """
    blocks = []
    
    # 1. Ensure coverage (Seed 24)
    for col in stratification_cols:
        try:
            # Group by column and take 1 sample
            sample = df.groupby(col, group_keys=False).apply(lambda x: x.sample(n=1, random_state=24))
            blocks.append(sample)
        except Exception as e:
            st.warning(f"Could not stratify by column '{col}': {e}")

    if not blocks:
        return pd.DataFrame(), 0

    # Combine blocks to form the base sample
    current_sample = pd.concat(blocks).drop_duplicates()
    guaranteed_count = len(current_sample)

    # 2. Tech Boost (Ensure Minimum per Tech)
    if col_tech and min_tech_count > 0 and col_tech in df.columns:
        unique_techs = df[col_tech].unique()
        
        for tech in unique_techs:
            # Check how many we already have for this tech
            current_tech_rows = current_sample[current_sample[col_tech] == tech]
            count_so_far = len(current_tech_rows)
            
            if count_so_far < min_tech_count:
                needed = min_tech_count - count_so_far
                
                # Find available rows for this tech NOT yet sampled
                available_indices = df.index.difference(current_sample.index)
                tech_pool = df.loc[available_indices]
                tech_pool = tech_pool[tech_pool[col_tech] == tech]
                
                if not tech_pool.empty:
                    # Take what is needed (or all if not enough exist)
                    take_n = min(needed, len(tech_pool))
                    tech_boost_sample = tech_pool.sample(n=take_n, random_state=42)
                    current_sample = pd.concat([current_sample, tech_boost_sample])

    # 3. Fill remainder (Seed 52)
    # Note: If Tech Boost pushed us over target_count, we keep the extra rows to satisfy the tech requirement.
    current_total = len(current_sample)
    target_count = int(target_count)
    remaining = target_count - current_total
    
    if remaining > 0:
        # Fill strictly from unsampled data
        available_indices = df.index.difference(current_sample.index)
        pool = df.loc[available_indices]
        
        n_sample = min(remaining, len(pool))
        if n_sample > 0:
            extra_sample = pool.sample(n=n_sample, random_state=52)
            current_sample = pd.concat([current_sample, extra_sample])
    
    # 4. Shuffle (Seed 99)
    final_sample = current_sample.sample(frac=1, random_state=99).reset_index(drop=True)
    
    return final_sample, guaranteed_count

def convert_df_to_excel(df):
    """Converts dataframe to Excel bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sampled Data')
    return output.getvalue()

# --- Logic: CGI Sampling ---

def calculate_cgi_metrics(df, value_cols):
    """CGI Logic: Strings first, else numeric average."""
    def process_row(row):
        values = row[value_cols]
        strings = [v for v in values if isinstance(v, str)]
        if strings:
            return strings[0]
        nums = [v for v in values if isinstance(v, (int, float)) and not pd.isna(v)]
        return sum(nums) / len(nums) if nums else None

    return df.apply(process_row, axis=1)

def categorize_cgi_results(df, result_col):
    """Bins: Best (0-0.1), Variable (0.1-1), Worst (>1)"""
    df["num_Result"] = pd.to_numeric(df[result_col], errors="coerce")
    bins = [0, 0.1, 1, float("inf")]
    labels = ["Best DCR", "Variable DCR", "Worst DCR"]
    
    df["Result_bin"] = pd.cut(df["num_Result"], bins=bins, labels=labels, include_lowest=True, right=True)
    df["Result_bin"] = df.apply(lambda row: row[result_col] if pd.isna(row["Result_bin"]) else row["Result_bin"], axis=1)
    df["Result_bin"] = df["Result_bin"].astype(object)
    return df

# --- Logic: Downtime Sampling ---

def calculate_downtime_counts(df, value_cols):
    """Downtime Logic: Count values > 0."""
    def process_row(row):
        subset = row[value_cols]
        cleaned_values = pd.to_numeric(subset, errors='coerce').fillna(0)
        return (cleaned_values > 0).sum()
    return df.apply(process_row, axis=1)

def categorize_downtime_counts(df, result_col):
    """Bins: 0-10, 10-20, 20-30, 30+"""
    df["num_Result"] = pd.to_numeric(df[result_col], errors="coerce")
    bins = [0, 10, 20, 30, float("inf")]
    labels = ["0–10", "10–20", "20–30", "30+"]
    df["Result_bin"] = pd.cut(df["num_Result"], bins=bins, labels=labels, include_lowest=True, right=False)
    df["Result_bin"] = df["Result_bin"].astype(object)
    df["Result_bin"] = df["Result_bin"].fillna("0–10")
    return df

# --- Main Application ---

def main():
    st.title("📊 Unified Sampling Tool")
    
    # --- Sidebar: Mode Selection ---
    st.sidebar.markdown("### ⚙️ Settings")
    mode = st.sidebar.radio("Select Sampling Mode", ["CGI Sampling", "Downtime Sampling"])
    
    st.sidebar.divider()
    
    # Display Mode Instructions
    if mode == "CGI Sampling":
        st.markdown("""
        **Selected: CGI Stratified Sampling**
        * **Logic**: Calculates Average of values.
        * **Min per Tech**: Ensures at least **385** rows per technology.
        * **Categories**: Best DCR, Variable DCR, Worst DCR.
        """)
        default_target = 2000
        # Enforce the 385 rule for CGI
        min_tech_req = 385
    else:
        st.markdown("""
        **Selected: Downtime Sampling**
        * **Logic**: Counts days with data > 0 (Blanks filled with 0).
        * **Categories**: 0–10, 10–20, 20–30, 30+.
        """)
        default_target = 1500
        # No specific tech minimum requested for Downtime
        min_tech_req = 0

    # 1. File Upload
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
    
    # --- Developer Info ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 Developer Info")
    st.sidebar.info("**Developed by:** Sudhir Jangra  \n**GitHub:** [github.com/sudhirjangra](https://github.com/sudhirjangra)")

    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
            with st.expander("👀 Preview Data", expanded=False):
                st.dataframe(df.head())

            # 2. Metadata Columns
            st.sidebar.subheader("1. Map Metadata")
            all_cols = df.columns.tolist()
            
            # Smart defaults
            idx_pin = next((i for i, c in enumerate(all_cols) if "pincode" in c.lower()), 1 if len(all_cols)>1 else 0)
            idx_dist = next((i for i, c in enumerate(all_cols) if "district" in c.lower()), 2 if len(all_cols)>2 else 0)
            idx_tech = next((i for i, c in enumerate(all_cols) if "tech" in c.lower()), 5 if len(all_cols)>5 else 0)

            col_pincode = st.sidebar.selectbox("Pincode", all_cols, index=idx_pin)
            col_district = st.sidebar.selectbox("District", all_cols, index=idx_dist)
            col_tech = st.sidebar.selectbox("Technology", all_cols, index=idx_tech)
            
            # 3. Value Columns
            st.sidebar.subheader("2. Map Data")
            default_start = 8 if len(all_cols) > 8 else 0
            start_col = st.sidebar.selectbox("Start Data Column", all_cols, index=default_start, help="Columns from here to the end will be used for calculation.")
            
            start_idx = all_cols.index(start_col)
            value_cols = all_cols[start_idx:]
            
            # 4. Target Size
            st.sidebar.subheader("3. Configuration")
            target_size = st.sidebar.number_input("Target Sample Size", min_value=100, max_value=len(df), value=default_target)
            
            # Allow user to override minimum if needed, but default to 385 for CGI
            if mode == "CGI Sampling":
                min_tech_req = st.sidebar.number_input("Min Samples per Technology", value=385)

            # --- RUN ---
            if st.button(f"🚀 Run {mode}", type="primary"):
                with st.spinner("Processing..."):
                    df_proc = df.copy()
                    
                    # Apply specific logic based on mode
                    if mode == "CGI Sampling":
                        df_proc["Result"] = calculate_cgi_metrics(df_proc, value_cols)
                        df_proc = categorize_cgi_results(df_proc, "Result")
                    else:
                        df_proc[value_cols] = df_proc[value_cols].fillna(0)
                        df_proc["Result"] = calculate_downtime_counts(df_proc, value_cols)
                        df_proc = categorize_downtime_counts(df_proc, "Result")
                    
                    # Stratified Sampling with Tech Minimums
                    strat_cols = [col_pincode, col_district, col_tech, "Result_bin"]
                    
                    sampled_df, guaranteed = perform_sampling(
                        df_proc, 
                        stratification_cols=strat_cols, 
                        target_count=target_size,
                        col_tech=col_tech,       # Pass tech column
                        min_tech_count=min_tech_req  # Pass minimum requirement (385 or 0)
                    )
                    
                    # Cleanup
                    if "num_Result" in sampled_df.columns:
                        sampled_df = sampled_df.drop(columns=["num_Result"])
                        
                st.success("Sampling Complete!")
                st.markdown("---")
                
                # --- Tabs for Output ---
                t1, t2, t3 = st.tabs(["📋 Summary", "📈 Visuals", "💾 Data Table"])
                
                with t1:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sample Size", len(sampled_df))
                    c2.metric("Guaranteed Coverage", guaranteed)
                    c3.metric("Original Size", len(df))
                    
                    st.write("### Technology Breakdown")
                    tech_counts = sampled_df[col_tech].value_counts().reset_index()
                    tech_counts.columns = ["Technology", "Count"]
                    st.dataframe(tech_counts, use_container_width=True)
                    
                    if mode == "CGI Sampling":
                        # Validation Check
                        below_min = tech_counts[tech_counts["Count"] < min_tech_req]
                        if not below_min.empty:
                            st.warning(f"⚠️ Some technologies have fewer than {min_tech_req} samples because the original file didn't have enough data for them.")
                        else:
                            st.success(f"✅ All technologies have at least {min_tech_req} samples.")
                    
                with t2:
                    st.write("### Distribution Comparison")
                    c1, c2 = st.columns(2)
                    
                    if mode == "Downtime Sampling":
                        order = ["0–10", "10–20", "20–30", "30+"]
                        pal = "magma"
                    else:
                        order = None
                        pal = "viridis"

                    with c1:
                        st.write("**Original Distribution**")
                        fig1, ax1 = plt.subplots(figsize=(6,4))
                        sns.countplot(y=df_proc["Result_bin"], ax=ax1, order=order, palette=pal)
                        st.pyplot(fig1)
                        
                    with c2:
                        st.write("**Sampled Distribution**")
                        fig2, ax2 = plt.subplots(figsize=(6,4))
                        sns.countplot(y=sampled_df["Result_bin"], ax=ax2, order=order, palette=pal)
                        st.pyplot(fig2)
                        
                with t3:
                    st.dataframe(sampled_df, use_container_width=True)
                    if mode == "Downtime Sampling":
                        st.caption("✅ **Note:** Blank cells in data columns filled with **0**.")
                    else:
                        st.caption("Original column names preserved.")
                    
                # --- Download ---
                st.markdown("---")
                clean_name = uploaded_file.name.rsplit('.', 1)[0]
                out_name = f"Sampled_{mode.split()[0]}_{clean_name}.xlsx"
                excel_data = convert_df_to_excel(sampled_df)
                
                st.download_button(
                    label="📥 Download Sampled Excel",
                    data=excel_data,
                    file_name=out_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

    else:
        st.info("Please upload an Excel file to begin.")

if __name__ == "__main__":
    main()