import streamlit as st
import pandas as pd
import os
import io

st.set_page_config(page_title="DCR–Downtime Sampling", layout="wide")
st.title("DCR – Downtime Sampling")


col1, col2 = st.columns([2, 2])

with col1:
    dcr_file = st.file_uploader("Upload DCR Sheet", type=["xlsx"])

with col2:
    downtime_file = st.file_uploader("Upload Downtime Sheet", type=["xlsx"])

col3, col4 = st.columns([2, 2])

with col3:
    percent = st.number_input("Replace Percentage", min_value=1, max_value=99, value=35, step=1)

with col4:
    random_state = st.number_input("Random Seed (Optional)", min_value=1, value=42, step=1)


def proportional_sample(df, group_col, n_samples, random_state):
    proportions = df[group_col].value_counts(normalize=True)
    samples = []

    for val, prop in proportions.items():
        k = int(round(prop * n_samples))
        subset = df[df[group_col] == val]

        if k > len(subset):
            k = len(subset)

        if k > 0:
            samples.append(subset.sample(k, random_state=random_state))

    return pd.concat(samples) if samples else pd.DataFrame(columns=df.columns)



if dcr_file and downtime_file and st.button("LFG"):

    df_dcr = pd.read_excel(dcr_file)
    df_dt = pd.read_excel(downtime_file)

    required_cols = {"Result_bin", "district"}
    if not required_cols.issubset(df_dcr.columns) or not required_cols.issubset(df_dt.columns):
        st.error("Both sheets must contain 'Result_bin' and 'district' columns")
        st.stop()

    total_rows = len(df_dcr)
    remove_count = int((percent / 100) * total_rows)

    original_districts = set(df_dcr["district"].unique())

    rows_removed = proportional_sample(
        df_dcr, "Result_bin", remove_count, random_state
    )

    df_dcr_remaining = df_dcr.drop(rows_removed.index)

    rows_inserted = proportional_sample(
        df_dt, "Result_bin", len(rows_removed), random_state
    )

    df_final = pd.concat([df_dcr_remaining, rows_inserted], ignore_index=True)

    missing_districts = original_districts - set(df_final["district"].unique())
    if missing_districts:
        restore_rows = (
            df_dcr[df_dcr["district"].isin(missing_districts)]
            .drop_duplicates("district")
        )
        df_final = pd.concat([df_final, restore_rows], ignore_index=True)




    tab1, tab2 = st.tabs(["Charts & Summary", "Data Preview"])

    with tab1:
        st.subheader("Result_bin Distribution")

        before_dist = df_dcr["Result_bin"].value_counts(normalize=True)
        after_dist = df_final["Result_bin"].value_counts(normalize=True)

        chart_df = pd.DataFrame({
            "Before (DCR)": before_dist,
            "After (Final)": after_dist
        }).fillna(0)

        st.bar_chart(chart_df)

        st.subheader("District Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Unique Districts (Before)", len(original_districts))
        with col_b:
            st.metric("Unique Districts (After)", df_final["district"].nunique())

    with tab2:
        st.subheader("Rows Removed from DCR")
        st.write(f"Total Removed: {len(rows_removed)}")
        st.dataframe(rows_removed.head(200))

        st.subheader("Rows Inserted from Downtime")
        st.write(f"Total Inserted: {len(rows_inserted)}")
        st.dataframe(rows_inserted.head(200))

        st.subheader("Final DCR Data")
        st.write(f"Final Row Count: {len(df_final)}")
        st.dataframe(df_final.head(300))       



    dcr_filename = os.path.splitext(dcr_file.name)[0]
    output_filename = f"{percent}_DT_{dcr_filename}.xlsx"

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_final.to_excel(writer, index=False)

    buffer.seek(0)

    st.download_button(
        label="Export xlsx",
        data=buffer,
        file_name=output_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
