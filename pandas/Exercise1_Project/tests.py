import pandas as pd
from main import (
    load_student_enquiries,
    clean_enquiry_data,
    merge_with_courses,
    add_net_quoted_fee,
    aggregate_by_course,
    finalize_summary,
)


def test_load_excludes_date(tmp_path):
    csv_file = tmp_path / "enquiries.csv"
    csv_file.write_text(
        "enquiry_id,course_id,student_name,email,phone,quoted_fee,discount_pct,enquiry_date\n"
        "3001,11,Aarav Rao,aarav@example.com,9876501011,30000,10,2026-09-01\n"
    )
    df = load_student_enquiries(str(csv_file))
    assert "enquiry_date" not in df.columns
    assert list(df.columns) == [
        "enquiry_id",
        "course_id",
        "student_name",
        "email",
        "phone",
        "quoted_fee",
        "discount_pct",
    ]


def test_clean_removes_missing_email_and_keeps_first_duplicate():
    raw_df = pd.DataFrame(
        {
            "enquiry_id": [3002, 3004, 3002],
            "course_id": [12, 14, 12],
            "student_name": ["Diya Sen", "Meera Nair", "Diya Sen Duplicate"],
            "email": ["diya@example.com", None, "diya2@example.com"],
            "phone": ["9876501012", "9876501014", "9876501099"],
            "quoted_fee": [24000, 18000, 25000],
            "discount_pct": [5, 10, 0],
        }
    )
    cleaned = clean_enquiry_data(raw_df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["student_name"] == "Diya Sen"
    assert cleaned.iloc[0]["quoted_fee"] == 24000


def test_merge_removes_unmatched_course_id(tmp_path):
    courses_file = tmp_path / "courses.csv"
    courses_file.write_text(
        "course_id,course_name,category,duration_months,standard_fee\n"
        "11,Data Analytics,Data,4,35000\n"
    )
    enquiries_df = pd.DataFrame(
        {
            "enquiry_id": [3001, 3008],
            "course_id": [11, 99],
            "student_name": ["Aarav Rao", "Nisha Paul"],
            "email": ["aarav@example.com", "nisha@example.com"],
            "phone": ["9876501011", "9876501018"],
            "quoted_fee": [30000, 26000],
            "discount_pct": [10, 10],
        }
    )
    merged = merge_with_courses(enquiries_df, str(courses_file))
    assert len(merged) == 1
    assert merged.iloc[0]["course_id"] == 11
    assert "course_name" in merged.columns


def test_net_fee_and_aggregation_and_final_summary():
    df = pd.DataFrame(
        {
            "enquiry_id": [3001, 3006, 3003],
            "course_id": [11, 11, 13],
            "course_name": ["Data Analytics", "Data Analytics", "Cloud Fundamentals"],
            "quoted_fee": [30000, 32000, 20000],
            "discount_pct": [10, 15, 0],
        }
    )
    with_net = add_net_quoted_fee(df)
    assert list(with_net["net_quoted_fee"]) == [27000.0, 27200.0, 20000.0]

    agg = aggregate_by_course(with_net)
    final = finalize_summary(agg)

    assert list(final.columns) == ["course_name", "avg_net_quoted_fee", "enquiry_count"]
    assert list(final["course_name"]) == ["Cloud Fundamentals", "Data Analytics"]
    assert list(final["avg_net_quoted_fee"]) == [20000.0, 27100.0]
    assert list(final["enquiry_count"]) == [1, 2]
    assert list(final.index) == [0, 1]