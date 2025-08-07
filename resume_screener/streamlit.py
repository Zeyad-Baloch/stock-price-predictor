import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from resume import ResumeScreener
from bias_detection import MLResumeScreener, BiasDetector

st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'screener' not in st.session_state:
    st.session_state.screener = ResumeScreener()
if 'ml_screener' not in st.session_state:
    st.session_state.ml_screener = MLResumeScreener(st.session_state.screener)
if 'bias_detector' not in st.session_state:
    st.session_state.bias_detector = BiasDetector(st.session_state.ml_screener)
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False


def train_ml_model():

    with st.spinner("Training ML model... This may take a moment."):
        progress_bar = st.progress(0)
        progress_bar.progress(20)

        results = st.session_state.ml_screener.train_model(num_samples=800)
        progress_bar.progress(80)

        st.session_state.screener.attach_ml_screener(st.session_state.ml_screener)
        progress_bar.progress(100)

        st.session_state.model_trained = True
        st.success(f"Model trained successfully!")


def process_files(uploaded_files):
    """Process uploaded resume files"""
    resumes = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_content = file.read()
            text = st.session_state.screener.extract_text_from_pdf(pdf_content)
        else:
            text = file.read().decode('utf-8')

        resumes.append(text)
    return resumes


def create_ranking_chart(rankings, title="Resume Rankings"):

    df = pd.DataFrame([
        {
            'Resume ID': i + 1,
            'Overall Score': result['match_results']['overall_score'],
            'Skills': result['match_results']['skill_score'],
            'Experience': result['match_results']['experience_score'],
            'Education': result['match_results']['education_score'],
            'Matched Skills': len(result['match_results']['matched_skills'])
        }
        for i, result in enumerate(rankings)
    ])

    fig = px.bar(df, x='Resume ID', y='Overall Score',
                 title=title,
                 color='Overall Score',
                 color_continuous_scale='RdYlGn')

    fig.update_layout(height=400)
    return fig, df


def create_comparison_chart(original_rankings, mitigated_rankings):

    comparison_data = []

    for i, (orig, mitig) in enumerate(zip(original_rankings, mitigated_rankings)):
        comparison_data.append({
            'Resume ID': i + 1,
            'Original Score': orig['match_results']['overall_score'],
            'Mitigated Score': mitig['match_results']['overall_score'],
            'Adjustment': mitig['match_results'].get('bias_adjustment', 0)
        })

    df = pd.DataFrame(comparison_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Original Score',
        x=df['Resume ID'],
        y=df['Original Score'],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='Mitigated Score',
        x=df['Resume ID'],
        y=df['Mitigated Score'],
        marker_color='darkgreen'
    ))

    fig.update_layout(
        title='Score Comparison: Before vs After Bias Mitigation',
        xaxis_title='Resume ID',
        yaxis_title='Score',
        barmode='group',
        height=500
    )

    return fig, df


def create_bias_analysis_chart(bias_report, analysis_df):
    """Create bias analysis visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Gender Score Distribution', 'Age Group Score Distribution',
                        'Demographic Parity', 'Score vs Rank'),
        specs=[[{"type": "box"}, {"type": "box"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    # Gender distribution
    gender_data = analysis_df[analysis_df['gender'] != 'unknown']
    if len(gender_data) > 0:
        for gender in gender_data['gender'].unique():
            gender_scores = gender_data[gender_data['gender'] == gender]['score']
            fig.add_trace(go.Box(y=gender_scores, name=gender, showlegend=False), row=1, col=1)

    # Age group distribution
    age_data = analysis_df[analysis_df['age_group'] != 'unknown']
    if len(age_data) > 0:
        for age_group in age_data['age_group'].unique():
            age_scores = age_data[age_data['age_group'] == age_group]['score']
            fig.add_trace(go.Box(y=age_scores, name=age_group, showlegend=False), row=1, col=2)

    # Demographic parity
    if 'demographic_parity' in bias_report:
        parity_data = bias_report['demographic_parity']
        if 'gender' in parity_data:
            rates = parity_data['gender']['rates']
            fig.add_trace(go.Bar(x=list(rates.keys()), y=list(rates.values()),
                                 name='Gender Parity', showlegend=False), row=2, col=1)

    # Score vs Rank
    fig.add_trace(go.Scatter(x=analysis_df['rank'], y=analysis_df['score'],
                             mode='markers', name='Resumes', showlegend=False), row=2, col=2)

    fig.update_layout(height=600, title_text="Bias Analysis Dashboard")
    return fig


def main():
    st.title("Resume Screener")
    st.markdown("Upload resumes and job descriptions to get score,rankings with bias detection")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Model training
        if not st.session_state.model_trained:
            if st.button("Train ML Model", type="primary"):
                train_ml_model()
        else:
            st.success("✅ ML Model Ready")

        # Bias mitigation settings
        st.subheader("Bias Mitigation")
        enable_bias_mitigation = st.checkbox("Enable Bias Mitigation", value=True)
        mitigation_strength = st.slider("Mitigation Strength", 0.0, 1.0, 0.5, 0.1)


    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Job Description")
        job_text = st.text_area("Enter job description:",
                                placeholder="Senior Python Developer with 5+ years experience...",
                                height=200)

    with col2:
        st.subheader("Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF or TXT):",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )

    if st.button("Analyze Resumes", type="primary", disabled=not (job_text and uploaded_files)):
        if not job_text:
            st.error("Please enter a job description")
            return
        if not uploaded_files:
            st.error("Please upload at least one resume")
            return

        with st.spinner("Processing resumes..."):
            # Process files
            resumes = process_files(uploaded_files)

            # Get rankings
            rankings = st.session_state.screener.rank_resumes(resumes, job_text)

            # Store original rankings for comparison
            original_rankings = [r.copy() for r in rankings]

            # Bias analysis
            bias_report, analysis_df = st.session_state.bias_detector.analyze_bias(
                resumes, job_text, rankings
            )

            # Apply bias mitigation if enabled
            if enable_bias_mitigation:
                mitigated_rankings = st.session_state.bias_detector.mitigate_bias(
                    rankings, bias_report, mitigation_strength
                )
            else:
                mitigated_rankings = rankings

        # Display results
        st.header(""
                  "Results")


        tab1, tab2, tab3, tab4 = st.tabs(["Rankings", "Comparison", "Bias Analysis", "Details"])

        with tab1:
            st.subheader("Final Rankings")
            ranking_fig, ranking_df = create_ranking_chart(mitigated_rankings)
            st.plotly_chart(ranking_fig, use_container_width=True)

            # Top candidates summary
            st.subheader("Top 3 Candidates")
            for i, result in enumerate(mitigated_rankings[:3]):
                with st.expander(f"Rank {i + 1} - Resume {result['resume_id'] + 1}"):
                    match = result['match_results']
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Overall Score", f"{match['overall_score']:.3f}")
                        st.metric("Skills Match", f"{match['skill_score']:.3f}")

                    with col2:
                        st.metric("Experience Match", f"{match['experience_score']:.3f}")
                        st.metric("Education Match", f"{match['education_score']:.3f}")

                    with col3:
                        st.metric("Matched Skills", len(match['matched_skills']))
                        if 'bias_adjustment' in match:
                            st.metric("Bias Adjustment", f"{match['bias_adjustment']:.3f}")

                    if match['matched_skills']:
                        st.write("**Matched Skills:**", ", ".join(match['matched_skills']))

        with tab2:
            st.subheader("Before vs After Mitigation")
            if enable_bias_mitigation:
                comparison_fig, comparison_df = create_comparison_chart(original_rankings, mitigated_rankings)
                st.plotly_chart(comparison_fig, use_container_width=True)


                st.subheader("Adjustment Details")
                st.dataframe(comparison_df.style.format({
                    'Original Score': '{:.3f}',
                    'Mitigated Score': '{:.3f}',
                    'Adjustment': '{:.3f}'
                }))
            else:
                st.info("Enable bias mitigation to see comparison")

        with tab3:
            st.subheader("Bias Analysis")

            # Bias metrics
            if bias_report:
                col1, col2 = st.columns(2)

                with col1:
                    if 'gender_test' in bias_report:
                        test_result = bias_report['gender_test']
                        st.metric(
                            "Gender Bias Test",
                            "Significant" if test_result['significant'] else "Not Significant",
                            f"p-value: {test_result['p_value']:.3f}"
                        )

                with col2:
                    if 'demographic_parity' in bias_report:
                        parity = bias_report['demographic_parity']
                        if 'gender' in parity:
                            max_diff = parity['gender']['max_difference']
                            st.metric("Gender Parity Gap", f"{max_diff:.3f}")


                bias_fig = create_bias_analysis_chart(bias_report, analysis_df)
                st.plotly_chart(bias_fig, use_container_width=True)
            else:
                st.info("No bias detected in current analysis")

        with tab4:
            st.subheader("Detailed Analysis")

            # Feature importance
            if st.session_state.model_trained:
                importance_df = st.session_state.ml_screener.get_feature_importance()
                if importance_df is not None:
                    st.subheader("Feature Importance")
                    importance_fig = px.bar(
                        importance_df.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    st.plotly_chart(importance_fig, use_container_width=True)


            st.subheader("All Rankings Data")
            detailed_df = pd.DataFrame([
                {
                    'Resume ID': result['resume_id'] + 1,
                    'Overall Score': result['match_results']['overall_score'],
                    'Skills Score': result['match_results']['skill_score'],
                    'Experience Score': result['match_results']['experience_score'],
                    'Education Score': result['match_results']['education_score'],
                    'Text Similarity': result['match_results']['text_similarity'],
                    'Matched Skills Count': len(result['match_results']['matched_skills']),
                    'Bias Adjustment': result['match_results'].get('bias_adjustment', 0)
                }
                for result in mitigated_rankings
            ])

            st.dataframe(detailed_df.style.format({
                col: '{:.3f}' for col in detailed_df.columns if col != 'Resume ID' and col != 'Matched Skills Count'
            }))


if __name__ == "__main__":
    main()