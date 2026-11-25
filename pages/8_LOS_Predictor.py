"""Page 8: Interactive Length of Stay Predictor."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="LOS Predictor | MediScope AI", page_icon="🤖", layout="wide")

from src.prediction import (
    load_model_metadata,
    load_trained_pipeline,
    predict_length_of_stay,
    validate_prediction_inputs,
)
from src.utils import inject_custom_css, render_kpi_card, render_page_header


@st.cache_resource(show_spinner="Loading trained prediction pipeline...")
def _get_cached_model():
    return load_trained_pipeline(), load_model_metadata()


def main() -> None:
    inject_custom_css()
    render_page_header(
        "Interactive Length-of-Stay Predictor",
        "Operational inpatient bed-day forecasting powered by the validated MediScope AI regression pipeline.",
    )

    try:
        pipeline, meta = _get_cached_model()
    except FileNotFoundError as exc:
        st.error(f"Prediction Unavailable: {exc}")
        st.info(
            "Please train the machine learning pipeline locally (`python -m src.train_model`) "
            "so `models/los_model_pipeline.joblib` and `models/model_metadata.json` are present."
        )
        return

    opts = meta["feature_options"]
    diag_list = opts["diagnoses"]
    diag_labels = [f"{d['code']} — {d['description']}" for d in diag_list]
    label_to_diag = {f"{d['code']} — {d['description']}": d for d in diag_list}

    sev_labels = {
        "1": "1 — Minor",
        "2": "2 — Moderate",
        "3": "3 — Major",
        "4": "4 — Extreme",
        "0": "0 — Not Applicable",
    }

    with st.form("los_predictor_form"):
        st.markdown("#### Enter Patient Admission & Clinical Acuity Attributes")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_group = st.selectbox("Age Group", options=opts["age_group"], index=3)
            gender = st.selectbox("Gender", options=opts["gender"], index=0)
            type_of_admission = st.selectbox("Type of Admission", options=opts["type_of_admission"], index=0)
        with c2:
            ed_ind = st.selectbox(
                "Emergency Department Indicator",
                options=opts["emergency_department_indicator"],
                format_func=lambda x: "Yes (ED Admission)" if x == "Y" else "No (Direct / Elective)",
            )
            med_surg = st.selectbox(
                "Medical / Surgical Category",
                options=opts["apr_medical_surgical_description"],
            )
            sev_code = st.selectbox(
                "APR Severity of Illness",
                options=opts["apr_severity_code"],
                format_func=lambda x: sev_labels.get(x, x),
                index=min(1, len(opts["apr_severity_code"]) - 1),
            )
        with c3:
            risk_mort = st.selectbox(
                "APR Risk of Mortality",
                options=opts["apr_risk_of_mortality"],
                index=0,
            )
            selected_diag_label = st.selectbox(
                "Primary CCSR Diagnosis",
                options=diag_labels,
                index=0,
            )

        submitted = st.form_submit_button("Predict Length of Stay", type="primary", width='stretch')

    if submitted:
        diag_info = label_to_diag[selected_diag_label]
        payload = {
            "age_group": age_group,
            "gender": gender,
            "type_of_admission": type_of_admission,
            "emergency_department_indicator": ed_ind,
            "ccsr_diagnosis_code": diag_info["code"],
            "apr_mdc_code": diag_info["mdc_code"],
            "apr_medical_surgical_description": med_surg,
            "apr_severity_code": sev_code,
            "apr_risk_of_mortality": risk_mort,
        }

        is_valid, errors = validate_prediction_inputs(payload, meta)
        if not is_valid:
            st.error("Input Validation Failed: " + "; ".join(errors))
            return

        pred_days = predict_length_of_stay(payload, pipeline=pipeline, metadata=meta)

        st.success("Prediction generated successfully using the validated pipeline.")
        r1, r2, r3 = st.columns(3)
        with r1:
            render_kpi_card(
                "Predicted Length of Stay",
                f"{pred_days:.1f} Days",
                f"Rounded operational estimate: {round(pred_days)} day(s)",
            )
        with r2:
            render_kpi_card(
                "Selected Diagnosis",
                diag_info["code"],
                diag_info["description"][:48],
                navy=True,
            )
        with r3:
            render_kpi_card(
                "Model Used",
                meta["selected_model"].replace(" Regressor", ""),
                f"MDC {diag_info['mdc_code']}: {diag_info['mdc_description'][:36]}",
            )

        st.markdown(
            """
            <div class="insight-box">
                <strong>Important Disclaimer:</strong> This prediction is a statistical estimate derived from
                historical 2021 New York State SPARCS inpatient discharges to support hospital bed planning,
                staffing, and operational scheduling. It is <strong>not</strong> a clinical decision, medical
                diagnosis, or treatment recommendation.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
