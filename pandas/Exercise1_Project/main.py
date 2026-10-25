import pandas as pd


def load_student_enquiries(file_path: str) -> pd.DataFrame:
    """Read training_enquiries.csv and return required columns excluding enquiry_date."""
    required_cols = [
        "enquiry_id",
        "course_id",
        "student_name",
        "email",
        "phone",
        "quoted_fee",
        "discount_pct",
    ]
    df = pd.read_csv(file_path)
    return df[required_cols].copy()


def clean_enquiry_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing email and keep the first occurrence of duplicate enquiry_ids."""
    cleaned = df.copy()
    # Drop NaN/null emails and empty/whitespace-only email strings
    cleaned = cleaned.dropna(subset=["email"])
    cleaned = cleaned[cleaned["email"].astype(str).str.strip() != ""]
    # Keep only the first occurrence of each enquiry_id in input order
    cleaned = cleaned.drop_duplicates(subset=["enquiry_id"], keep="first")
    return cleaned


def merge_with_courses(enquiries_df: pd.DataFrame, ref_file_path: str) -> pd.DataFrame:
    """Inner-join enquiries with training_courses.csv on course_id."""
    courses_df = pd.read_csv(ref_file_path)
    course_cols = [
        "course_id",
        "course_name",
        "category",
        "duration_months",
        "standard_fee",
    ]
    merged = pd.merge(
        enquiries_df,
        courses_df[course_cols],
        on="course_id",
        how="inner",
    )
    return merged


def add_net_quoted_fee(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate net_quoted_fee = quoted_fee * (1 - discount_pct / 100) rounded to 2 decimals."""
    result = df.copy()
    result["net_quoted_fee"] = (
        result["quoted_fee"] * (1 - result["discount_pct"] / 100.0)
    ).round(2)
    return result


def aggregate_by_course(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average net_quoted_fee and valid enquiry count per course."""
    aggregated = (
        df.groupby("course_name", as_index=False)
        .agg(
            avg_net_quoted_fee=("net_quoted_fee", "mean"),
            enquiry_count=("enquiry_id", "count"),
        )
        .reset_index(drop=True)
    )
    return aggregated


def finalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns, round avg_net_quoted_fee to 2 decimals, sort by course_name, and reset index."""
    final_df = df.copy()
    final_df["avg_net_quoted_fee"] = final_df["avg_net_quoted_fee"].round(2)
    final_df = final_df.sort_values(by="course_name", ascending=True).reset_index(drop=True)
    return final_df[["course_name", "avg_net_quoted_fee", "enquiry_count"]]


if __name__ == "__main__":
    enquiries_path = "data/training_enquiries.csv"
    courses_path = "data/training_courses.csv"

    step1_loaded = load_student_enquiries(enquiries_path)
    step2_cleaned = clean_enquiry_data(step1_loaded)
    step3_merged = merge_with_courses(step2_cleaned, courses_path)
    step4_with_net = add_net_quoted_fee(step3_merged)
    step5_aggregated = aggregate_by_course(step4_with_net)
    step6_final = finalize_summary(step5_aggregated)

    print(step6_final)