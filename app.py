"""
JobShield — Streamlit Application

Fake Job Posting Detection Using Deep Neural Networks
"""

import streamlit as st

from src.predict import (
    load_artifacts,
    predict_job
)


st.set_page_config(
    page_title="JobShield | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def get_model_artifacts():
    """
    Load the model once and reuse it
    during application interactions.
    """
    return load_artifacts()


def main():

    st.title("🛡️ JobShield")

    st.subheader(
        "Fake Job Posting Detection Using "
        "Deep Neural Networks"
    )

    st.write(
        "Analyse a job advertisement using a "
        "trained Deep Neural Network."
    )

    st.info(
        "JobShield is an academic screening tool. "
        "Its predictions do not establish whether "
        "an advertisement is genuinely fraudulent."
    )

    # Load saved model artefacts
    try:
        artifacts = get_model_artifacts()

    except Exception as error:
        st.error("Unable to load JobShield model.")
        st.exception(error)
        st.stop()

    with st.sidebar:

        st.header("About JobShield")

        st.write(
            "JobShield uses text and structured "
            "job information to screen advertisements "
            "for potentially fraudulent patterns."
        )

        st.markdown("**Model:** Deep Neural Network")
        st.markdown("**Framework:** TensorFlow / Keras")
        st.markdown("**Dataset:** EMSCAD")

        st.metric(
            "Classification Threshold",
            f"{artifacts['threshold']:.3f}"
        )

        st.caption(
            "The threshold was selected using "
            "validation data."
        )

        st.divider()

        st.warning(
            "The sigmoid output is not a calibrated "
            "fraud probability. Always verify "
            "suspicious job advertisements independently."
        )

    st.divider()

    st.header("Analyse a Job Posting")

    st.caption(
        "Enter the main information below. "
        "Additional fields are optional."
    )

    with st.form("jobshield_form"):

        title = st.text_input(
            "Job Title",
            placeholder="e.g. Software Developer"
        )

        description = st.text_area(
            "Paste Job Description",
            height=180,
            placeholder=(
                "Paste the full job advertisement here..."
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            company_profile = st.text_area(
                "Company Profile",
                height=100
            )

        with col2:
            requirements = st.text_area(
                "Job Requirements",
                height=100
            )

        benefits = st.text_area(
            "Benefits",
            height=80
        )

        with st.expander(
            "Additional Job Information (Optional)"
        ):

            left, right = st.columns(2)

            with left:

                location = st.text_input(
                    "Location"
                )

                department = st.text_input(
                    "Department"
                )

                salary_range = st.text_input(
                    "Salary Range"
                )

                employment_type = st.text_input(
                    "Employment Type",
                    placeholder="e.g. Full-time"
                )

            with right:

                required_experience = st.text_input(
                    "Required Experience"
                )

                required_education = st.text_input(
                    "Required Education"
                )

                industry = st.text_input(
                    "Industry"
                )

                function = st.text_input(
                    "Job Function"
                )

            st.markdown("**Binary Job Attributes**")

            telecommuting = st.checkbox(
                "Telecommuting / Remote Work"
            )

            has_company_logo = st.checkbox(
                "Company Logo Present"
            )

            has_questions = st.checkbox(
                "Screening Questions Present"
            )

            st.caption(
                "Unchecked binary attributes are "
                "encoded as 0."
            )

        submitted = st.form_submit_button(
            "Analyse Job Posting",
            type="primary",
            use_container_width=True
        )

    if submitted:

        job_data = {
            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "location": location,
            "department": department,
            "salary_range": salary_range,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,
            "industry": industry,
            "function": function,
            "telecommuting": int(telecommuting),
            "has_company_logo": int(has_company_logo),
            "has_questions": int(has_questions)
        }

        try:

            result = predict_job(
                job_data,
                artifacts
            )

        except ValueError as error:
            st.warning(str(error))

        except Exception as error:
            st.error(
                "An error occurred during prediction."
            )
            st.exception(error)

        else:

            st.divider()

            st.header("JobShield Analysis Result")

            if result["flagged"]:

                st.error(
                    "⚠️ POTENTIALLY FRAUDULENT"
                )

                st.write(
                    "The model has flagged this posting "
                    "for further verification."
                )

            else:

                st.success(
                    "✅ NOT FLAGGED AS FRAUDULENT"
                )

                st.write(
                    "The posting is below the selected "
                    "fraud-screening threshold. "
                    "This does not guarantee legitimacy."
                )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Estimated Fraud Score",
                    f"{result['score'] * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Screening Level",
                    result["risk_level"]
                )

            with col3:

                st.metric(
                    "Decision Threshold",
                    f"{result['threshold']:.3f}"
                )

            st.progress(
                min(max(result["score"], 0.0), 1.0)
            )

            st.caption(
                "The estimated fraud score is an "
                "uncalibrated neural-network output, "
                "not a verified probability of fraud."
            )

            st.caption(
                "LOW, MEDIUM and HIGH are illustrative "
                "screening bands derived from the "
                "decision threshold. They have not "
                "been separately validated."
            )

            st.info(
                "Recommendation: Verify the employer "
                "through its official website and "
                "trusted contact channels. Never "
                "send money or sensitive personal "
                "information solely on the basis "
                "of a job advertisement."
            )

    st.divider()

    st.caption(
        "JobShield | University Deep Learning Project | "
        "Educational Demonstration"
    )


if __name__ == "__main__":
    main()