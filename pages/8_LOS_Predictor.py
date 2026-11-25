"""Page 8: Interactive Length of Stay Predictor with Quick Presets."""

from __future__ import annotations

import streamlit as st

if not st.session_state.get("_nav_shell_active"):
    st.set_page_config(page_title="LOS Predictor | MediScope AI", page_icon="assets/mediscope_icon.svg", layout="wide")

from src.prediction import (
    load_model_metadata,
    load_trained_pipeline,
    predict_length_of_stay,
    validate_prediction_inputs,
)
from src.utils import (
    inject_custom_css,
    render_kpi_card,
    render_page_header,
    render_plain_english_guide,
)


@st.cache_resource(show_spinner="Loading trained prediction pipeline...")
def _get_cached_model():
    return load_trained_pipeline(), load_model_metadata()


def _apply_preset(preset: dict[str, str], diag_labels: list[str]) -> None:
    """Populate predictor form session keys cleanly from a clinical preset."""
    st.session_state["pred_age"] = preset["age_group"]
    st.session_state["pred_gender"] = preset["gender"]
    st.session_state["pred_adm"] = preset["type_of_admission"]
    st.session_state["pred_ed"] = preset["emergency_department_indicator"]
    st.session_state["pred_ms"] = preset["apr_medical_surgical_description"]
    st.session_state["pred_sev"] = preset["apr_severity_code"]
    st.session_state["pred_risk"] = preset["apr_risk_of_mortality"]
    prefix = preset["diag_code"] + " — "
    matched = [lbl for lbl in diag_labels if lbl.startswith(prefix)]
    if matched:
        st.session_state["pred_diag"] = matched[0]


def main() -> None:
    if not st.session_state.get("_nav_shell_active"):
        inject_custom_css()

    render_page_header(
        "Interactive Length-of-Stay Predictor",
        "Operational bed-day forecasting powered by the validated MediScope AI CatBoost regression pipeline.",
    )

    try:
        pipeline, meta = _get_cached_model()
    except FileNotFoundError as exc:
        st.error(f"Prediction Unavailable: {exc}")
        st.info(
            "Run `python -m src.train_model` locally so `models/los_model_pipeline.joblib` "
            "and `models/model_metadata.json` are generated."
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

    render_plain_english_guide(
        "How to Use the Predictor",
        "Choose a <strong>Quick Example Preset</strong> below or customize the patient's admission profile manually, then click "
        "<strong>Predict Length of Stay</strong>. The model outputs expected inpatient days to support hospital bed planning.",
    )

    # Quick clinical presets for easy exploration
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Preset: Routine Newborn (Minor)", width="stretch"):
            _apply_preset(
                {
                    "age_group": "0 to 17",
                    "gender": "F",
                    "type_of_admission": "Newborn",
                    "emergency_department_indicator": "N",
                    "apr_medical_surgical_description": "Medical",
                    "apr_severity_code": "1",
                    "apr_risk_of_mortality": "Minor",
                    "diag_code": "PNL001",
                },
                diag_labels,
            )
    with p2:
        if st.button("Preset: Elderly Septicemia (Major)", width="stretch"):
            _apply_preset(
                {
                    "age_group": "70 or Older",
                    "gender": "M",
                    "type_of_admission": "Emergency",
                    "emergency_department_indicator": "Y",
                    "apr_medical_surgical_description": "Medical",
                    "apr_severity_code": "3",
                    "apr_risk_of_mortality": "Major",
                    "diag_code": "INF002",
                },
                diag_labels,
            )
    with p3:
        if st.button("Preset: Acute Leukemia (Extreme)", width="stretch"):
            _apply_preset(
                {
                    "age_group": "50 to 69",
                    "gender": "M",
                    "type_of_admission": "Emergency",
                    "emergency_department_indicator": "Y",
                    "apr_medical_surgical_description": "Medical",
                    "apr_severity_code": "4",
                    "apr_risk_of_mortality": "Extreme",
                    "diag_code": "NEO060",
                },
                diag_labels,
            )

    # Initialize default form keys cleanly on first visit
    st.session_state.setdefault("pred_age", opts["age_group"][min(3, len(opts["age_group"]) - 1)])
    st.session_state.setdefault("pred_gender", opts["gender"][0])
    st.session_state.setdefault("pred_adm", "Emergency" if "Emergency" in opts["type_of_admission"] else opts["type_of_admission"][0])
    st.session_state.setdefault("pred_ed", "Y" if "Y" in opts["emergency_department_indicator"] else opts["emergency_department_indicator"][0])
    st.session_state.setdefault("pred_ms", opts["apr_medical_surgical_description"][0])
    st.session_state.setdefault("pred_sev", "2" if "2" in opts["apr_severity_code"] else opts["apr_severity_code"][0])
    st.session_state.setdefault("pred_risk", "Moderate" if "Moderate" in opts["apr_risk_of_mortality"] else opts["apr_risk_of_mortality"][0])
    st.session_state.setdefault("pred_diag", diag_labels[0])

    with st.form("los_predictor_form"):
        st.markdown("#### Patient Admission & Clinical Acuity Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_group = st.selectbox("Age Group", options=opts["age_group"], key="pred_age")
            gender = st.selectbox("Gender", options=opts["gender"], key="pred_gender")
            type_of_admission = st.selectbox("Type of Admission", options=opts["type_of_admission"], key="pred_adm")
        with c2:
            ed_ind = st.selectbox(
                "Emergency Department Indicator",
                options=opts["emergency_department_indicator"],
                format_func=lambda x: "Yes (ED Admission)" if x == "Y" else "No (Direct / Elective)",
                key="pred_ed",
            )
            med_surg = st.selectbox(
                "Medical / Surgical Category",
                options=opts["apr_medical_surgical_description"],
                key="pred_ms",
            )
            sev_code = st.selectbox(
                "APR Severity of Illness",
                options=opts["apr_severity_code"],
                format_func=lambda x: sev_labels.get(x, x),
                key="pred_sev",
            )
        with c3:
            risk_mort = st.selectbox(
                "APR Risk of Mortality",
                options=opts["apr_risk_of_mortality"],
                key="pred_risk",
            )
            selected_diag_label = st.selectbox(
                "Primary CCSR Diagnosis",
                options=diag_labels,
                key="pred_diag",
            )

        submitted = st.form_submit_button("Predict Length of Stay", type="primary", width="stretch")

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
        tier = (
            "Short Stay (1–3 days)"
            if pred_days < 3.5
            else ("Standard Inpatient (4–7 days)" if pred_days < 7.5 else "Extended Stay (8+ days)")
        )

        st.success("Prediction generated successfully using the validated pipeline.")
        r1, r2, r3 = st.columns(3)
        with r1:
            render_kpi_card(
                "Predicted Length of Stay",
                f"{pred_days:.1f} Days",
                f"Capacity Tier: {tier}",
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
                "Statewide Context (vs. Median 3.0d)",
                f"{pred_days - 3.0:+.1f}d vs. Median",
                f"Model: {meta['selected_model'].replace(' Regressor', '')}",
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


main()
