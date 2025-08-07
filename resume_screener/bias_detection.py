import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from resume import ResumeScreener

class MLResumeScreener:
    def __init__(self, base_screener):

        self.base_screener = base_screener
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False


    def extract_ml_features(self, resume, job):
        """Extract comprehensive features for ML model """
        skill_score = self.base_screener.calculate_skill_match_score(resume['skills'], job['required_skills'])
        experience_score = self.base_screener.calculate_experience_match_score(resume['experience_years'],
                                                                               job['required_experience'])
        education_score = self.base_screener.calculate_education_match_score(resume['education_level'],
                                                                             job['required_education'])
        text_similarity = self.base_screener.calculate_text_similarity(resume['cleaned_text'], job['cleaned_text'])

        basic_scores = {
            'skill_score': skill_score,
            'experience_score': experience_score,
            'education_score': education_score,
            'text_similarity': text_similarity
        }

        job_skills = set(job['required_skills'])
        resume_skills = set(resume['skills'])
        skill_intersection = job_skills.intersection(resume_skills)


        skill_overlap_ratio = len(skill_intersection) / len(job_skills) if job_skills else 0
        skill_coverage = len(skill_intersection) / len(resume_skills) if resume_skills else 0


        exp_ratio = resume['experience_years'] / max(1, job['required_experience'])
        exp_excess = max(0, resume['experience_years'] - job['required_experience']) / 10.0
        exp_deficit = max(0, job['required_experience'] - resume['experience_years']) / 10.0

        features = {
            # Basic matching features
            'skill_score': basic_scores['skill_score'],
            'experience_score': basic_scores['experience_score'],
            'education_score': basic_scores['education_score'],
            'text_similarity': basic_scores['text_similarity'],

            # skill features
            'total_skills': min(len(resume['skills']) / 15.0, 1.0),
            'skill_overlap_count': len(skill_intersection) / len(job_skills) if job_skills else 0,
            'skill_overlap_ratio': skill_overlap_ratio,
            'skill_coverage': skill_coverage,
            'skill_diversity': min(len(resume_skills) / 20.0, 1.0),
            'skill_specificity': skill_overlap_ratio * skill_coverage,
            # experience features
            'experience_years_norm': min(resume['experience_years'] / 15.0, 1.0),
            'experience_excess': exp_excess,
            'experience_ratio': min(exp_ratio, 2.0),
            'experience_deficit': exp_deficit,
            'experience_match': 1.0 if resume['experience_years'] >= job['required_experience'] else 0.0,

            # education features
            'education_numeric': self._encode_education(resume['education_level']) / 4.0,
            'education_match': 1.0 if self._encode_education(resume['education_level']) >= self._encode_education(
                job['required_education']) else 0.0,
            'education_overqualified': max(0,
                                           self._encode_education(resume['education_level']) - self._encode_education(
                                               job['required_education'])) / 4.0,

            'resume_length': min(len(resume['cleaned_text'].split()) / 200.0, 1.0),
            'keyword_density': self._calculate_keyword_density(resume['cleaned_text'], job['required_skills']),

            'overall_qualification': (basic_scores['skill_score'] + basic_scores['experience_score'] + basic_scores[
                'education_score']) / 3.0,
            'skill_exp_balance': skill_overlap_ratio * min(exp_ratio, 1.0),
        }

        return features
    def _calculate_keyword_density(self, text, keywords):

        text_lower = text.lower()
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0.0

        keyword_count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return keyword_count / total_words
    def _encode_education(self, education_level):

        education_mapping = {
            'unknown': 0,
            'diploma': 1,
            'bachelors': 2,
            'masters': 3,
            'phd': 4
        }
        return education_mapping.get(education_level, 0)


    def generate_synthetic_training_data(self, num_samples=1000):

        training_data = []


        job_templates = [
            {
                'title': 'Senior Python Developer',
                'required_skills': ['python', 'django', 'sql', 'git', 'aws'],
                'required_experience': 5,
                'required_education': 'bachelors'
            },
            {
                'title': 'Junior Frontend Developer',
                'required_skills': ['javascript', 'react', 'html', 'css'],
                'required_experience': 1,
                'required_education': 'bachelors'
            },
            {
                'title': 'Data Scientist',
                'required_skills': ['python', 'tensorflow', 'sql', 'analytical'],
                'required_experience': 3,
                'required_education': 'masters'
            },
            {
                'title': 'DevOps Engineer',
                'required_skills': ['docker', 'kubernetes', 'aws', 'jenkins'],
                'required_experience': 4,
                'required_education': 'bachelors'
            },
            {
                'title': 'Machine Learning Engineer',
                'required_skills': ['python', 'tensorflow', 'pytorch', 'sql', 'aws'],
                'required_experience': 4,
                'required_education': 'masters'
            },
            {
                'title': 'Full Stack Developer',
                'required_skills': ['javascript', 'react', 'nodejs', 'mongodb', 'git'],
                'required_experience': 3,
                'required_education': 'bachelors'
            },
            {
                'title': 'Software Architect',
                'required_skills': ['python', 'microservices', 'aws', 'docker', 'sql'],
                'required_experience': 8,
                'required_education': 'masters'
            },
            {
                'title': 'Mobile Developer',
                'required_skills': ['java', 'kotlin', 'android', 'git'],
                'required_experience': 3,
                'required_education': 'bachelors'
            }
        ]


        all_skills = ['python', 'javascript', 'java', 'react', 'django', 'flask', 'nodejs',
                      'sql', 'mongodb', 'postgresql', 'git', 'aws', 'docker', 'kubernetes',
                      'tensorflow', 'pytorch', 'analytical', 'html', 'css', 'jenkins',
                      'microservices', 'redis', 'elasticsearch', 'angular', 'vue', 'kotlin', 'android']

        for _ in range(num_samples):
            job_template = np.random.choice(job_templates)


            job_skills = set(job_template['required_skills'])
            overlap_probability = np.random.beta(2, 2)
            overlap_count = int(overlap_probability * len(job_skills))

            candidate_skills = list(np.random.choice(list(job_skills), size=overlap_count, replace=False))


            related_skills = {
                'python': ['django', 'flask', 'tensorflow', 'pytorch'],
                'javascript': ['react', 'nodejs', 'angular', 'vue'],
                'java': ['kotlin', 'android'],
                'aws': ['docker', 'kubernetes'],
                'sql': ['postgresql', 'mongodb']
            }

            for skill in candidate_skills:
                if skill in related_skills:
                    for related in related_skills[skill]:
                        if related not in candidate_skills and np.random.random() < 0.3:
                            candidate_skills.append(related)


            remaining_skills = [s for s in all_skills if s not in candidate_skills]
            additional_count = np.random.randint(1, 5)
            if additional_count > 0 and len(remaining_skills) > 0:
                additional_skills = list(np.random.choice(remaining_skills,
                                                          size=min(additional_count, len(remaining_skills)),
                                                          replace=False))
                candidate_skills.extend(additional_skills)


            required_exp = job_template['required_experience']
            exp_variance = np.random.normal(0, 1.5)
            experience_years = max(0, int(required_exp + exp_variance))


            education_levels = ['diploma', 'bachelors', 'masters', 'phd']
            education_weights = [0.15, 0.50, 0.30, 0.05]
            education_level = np.random.choice(education_levels, p=education_weights)

            candidate = {
                'skills': candidate_skills,
                'experience_years': experience_years,
                'education_level': education_level,
                'cleaned_text': f"Candidate with {len(candidate_skills)} skills and {experience_years} years experience. " + " ".join(
                    candidate_skills)
            }

            job = {
                'required_skills': job_template['required_skills'],
                'required_experience': job_template['required_experience'],
                'required_education': job_template['required_education'],
                'cleaned_text': f"Job requiring {len(job_template['required_skills'])} skills: " + " ".join(
                    job_template['required_skills'])
            }


            skill_overlap = len(set(candidate_skills) & set(job_template['required_skills']))
            total_required_skills = len(job_template['required_skills'])
            skill_match_ratio = skill_overlap / total_required_skills


            exp_diff = experience_years - required_exp
            if exp_diff >= 0:
                exp_score = min(1.0, 0.8 + (exp_diff * 0.03))
            else:
                exp_score = max(0.2, 0.8 + (exp_diff * 0.1))


            edu_levels = {'diploma': 1, 'bachelors': 2, 'masters': 3, 'phd': 4}
            candidate_edu = edu_levels.get(education_level, 1)
            required_edu = edu_levels.get(job_template['required_education'], 2)

            if candidate_edu >= required_edu:
                edu_score = min(1.0, 0.8 + (candidate_edu - required_edu) * 0.05)
            else:
                edu_score = max(0.3, 0.8 - (required_edu - candidate_edu) * 0.2)


            base_target = (
                    skill_match_ratio * 0.5 +
                    exp_score * 0.3 +
                    edu_score * 0.2
            )


            noise = np.random.normal(0, 0.05)
            target = np.clip(base_target + noise, 0.1, 0.9)


            if skill_overlap == total_required_skills and experience_years >= required_exp + 2:
                target = min(0.95, target + 0.1)

            features = self.extract_ml_features(candidate, job)
            training_data.append({
                'features': features,
                'target': target,
                'candidate': candidate,
                'job': job
            })

        return training_data

    def prepare_training_data(self, training_data):
        """Prepare features and targets for ML training"""
        features_list = []
        targets = []

        for sample in training_data:
            features_list.append(list(sample['features'].values()))
            targets.append(sample['target'])


        self.feature_names = list(training_data[0]['features'].keys())

        return np.array(features_list), np.array(targets)


    def train_model(self, num_samples=1000):

        print("Generating synthetic training data...")
        training_data = self.generate_synthetic_training_data(num_samples)

        print("Preparing features...")
        X, y = self.prepare_training_data(training_data)

        # Scaling features
        X_scaled = self.scaler.fit_transform(X)

        # Spliting data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )


        print("Training Random Forest model...")
        self.ml_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        self.ml_model.fit(X_train, y_train)

        # Evaluating model
        y_pred = self.ml_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        oob_score = self.ml_model.oob_score_

        print(f"Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"OOB Score: {oob_score:.4f}")

        # Feature importance analysis
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)


        top_2_importance = importance_df.head(2)['importance'].sum()
        if top_2_importance > 0.7:
            print(f"Warning: Top 2 features account for {top_2_importance:.1%} of importance - possible overfitting")

        self.is_trained = True
        return {'mse': mse, 'r2': r2, 'oob_score': oob_score}

    def validate_model(self, cv_folds=5):
        """Add cross-validation for better model assessment"""
        from sklearn.model_selection import cross_val_score

        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train_model() first.")

        validation_data = self.generate_synthetic_training_data(1000)
        X, y = self.prepare_training_data(validation_data)
        X_scaled = self.scaler.transform(X)

        cv_scores = cross_val_score(self.ml_model, X_scaled, y, cv=cv_folds, scoring='r2')

        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return cv_scores


    def predict_match_score(self, resume, job, use_ensemble=True):

        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train_model() first.")

        features = self.extract_ml_features(resume, job)
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)

        if use_ensemble:

            tree_predictions = []
            for tree in self.ml_model.estimators_[:min(50, len(self.ml_model.estimators_))]:
                pred = tree.predict(feature_vector_scaled)[0]
                tree_predictions.append(pred)


            prediction = np.median(tree_predictions)
        else:
            prediction = self.ml_model.predict(feature_vector_scaled)[0]


        return np.clip(prediction, 0.05, 0.95)
    def predict_match_score(self, resume, job):

        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train_model() first.")

        features = self.extract_ml_features(resume, job)
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)

        prediction = self.ml_model.predict(feature_vector_scaled)[0]
        return np.clip(prediction, 0, 1)

    def get_feature_importance(self):

        if not self.is_trained:
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df


class BiasDetector:
    def __init__(self, ml_screener):
        """Initialize bias detector with ML screener"""
        self.ml_screener = ml_screener


    def infer_demographics(self, resume_text):

        demographics = {
            'gender': 'unknown',
            'ethnicity': 'unknown',
            'age_group': 'unknown',
            'confidence': 0.0
        }

        text_lower = resume_text.lower()

        # gender inference
        male_indicators = ['mr.', 'he/him', 'his achievements', 'his experience', 'his role', 'his position']
        female_indicators = ['ms.', 'mrs.', 'she/her', 'her achievements', 'her experience', 'her role', 'her position']

        male_count = sum(1 for indicator in male_indicators if indicator in text_lower)
        female_count = sum(1 for indicator in female_indicators if indicator in text_lower)

        if male_count > female_count and male_count > 0:
            demographics['gender'] = 'male'
            demographics['confidence'] = min(male_count * 0.3, 1.0)
        elif female_count > male_count and female_count > 0:
            demographics['gender'] = 'female'
            demographics['confidence'] = min(female_count * 0.3, 1.0)

        # Age inference
        age_indicators = {
            'young': ['recent graduate', 'new grad', 'university 202', 'college 202', 'entry level', 'junior'],
            'experienced': ['senior', 'lead', 'principal', 'architect', 'manager', 'director'],
            'mid_career': ['5 years', '6 years', '7 years', '8 years', '9 years']
        }

        age_scores = {}
        for age_group, indicators in age_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                age_scores[age_group] = score

        if age_scores:
            demographics['age_group'] = max(age_scores.items(), key=lambda x: x[1])[0]
            demographics['confidence'] = max(demographics['confidence'],
                                             min(max(age_scores.values()) * 0.2, 1.0))

        return demographics


    def analyze_bias(self, resumes, job_posting, rankings):

        demographics = []
        for resume in resumes:
            if isinstance(resume, dict):
                demo = self.infer_demographics(resume['raw_text'])
            else:
                demo = self.infer_demographics(resume)
            demographics.append(demo)


        analysis_df = pd.DataFrame({
            'resume_id': range(len(resumes)),
            'rank': [i for i in range(len(rankings))],
            'score': [result['match_results']['overall_score'] for result in rankings],
            'gender': [d['gender'] for d in demographics],
            'age_group': [d['age_group'] for d in demographics],
            'ethnicity': [d['ethnicity'] for d in demographics]
        })

        bias_report = {}

        # Gender bias analysis
        if len(analysis_df[analysis_df['gender'] != 'unknown']) > 1:
            gender_stats = analysis_df[analysis_df['gender'] != 'unknown'].groupby('gender').agg({
                'score': ['mean', 'std', 'count'],
                'rank': ['mean']
            }).round(4)

            bias_report['gender_bias'] = gender_stats


            male_scores = analysis_df[analysis_df['gender'] == 'male']['score']
            female_scores = analysis_df[analysis_df['gender'] == 'female']['score']

            if len(male_scores) > 0 and len(female_scores) > 0:
                t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
                bias_report['gender_test'] = {
                    't_statistic': t_stat.item(),
                    'p_value': p_value.item(),
                    'significant': (p_value < 0.05).item()
                }

        # Age bias analysis
        if len(analysis_df[analysis_df['age_group'] != 'unknown']) > 1:
            age_stats = analysis_df[analysis_df['age_group'] != 'unknown'].groupby('age_group').agg({
                'score': ['mean', 'std', 'count'],
                'rank': ['mean']
            }).round(4)

            bias_report['age_bias'] = age_stats

        # Overall bias metrics
        bias_report['demographic_parity'] = self._calculate_demographic_parity(analysis_df)
        bias_report['equal_opportunity'] = self._calculate_equal_opportunity(analysis_df)

        return bias_report, analysis_df

    def _calculate_demographic_parity(self, df):

        top_50_percent = df['score'] >= df['score'].median()

        parity_metrics = {}

        for attr in ['gender', 'age_group']:
            if len(df[df[attr] != 'unknown']) > 1:
                group_rates = df[df[attr] != 'unknown'].groupby(attr).apply(
                    lambda x: (x['score'] >= df['score'].median()).mean()
                )
                parity_metrics[attr] = {
                    'rates': group_rates.to_dict(),
                    'max_difference': (group_rates.max() - group_rates.min()).item()
                }

        return parity_metrics

    def _calculate_equal_opportunity(self, df):

        qualified_threshold = df['score'].quantile(0.7)

        opportunity_metrics = {}

        for attr in ['gender', 'age_group']:
            if len(df[df[attr] != 'unknown']) > 1:
                qualified_rates = df[df[attr] != 'unknown'].groupby(attr).apply(
                    lambda x: (x['score'] >= qualified_threshold).mean()
                )
                opportunity_metrics[attr] = {
                    'rates': qualified_rates.to_dict(),
                    'max_difference': (qualified_rates.max() - qualified_rates.min()).item()
                }

        return opportunity_metrics

    def mitigate_bias(self, rankings, bias_report, mitigation_strength=0.5):
        """Apply bias mitigation to rankings"""
        if not bias_report:
            return rankings

        mitigated_rankings = []

        demographics = []
        original_scores = []

        for result in rankings:
            demo = self.infer_demographics(result['resume_data']['raw_text'])
            demographics.append(demo)
            original_scores.append(result['match_results']['overall_score'])

        group_stats = {}

        # Gender statistics
        if 'gender_bias' in bias_report:
            gender_groups = {}
            for i, demo in enumerate(demographics):
                gender = demo['gender']
                if gender != 'unknown':
                    if gender not in gender_groups:
                        gender_groups[gender] = []
                    gender_groups[gender].append(original_scores[i])

            if len(gender_groups) > 1:
                overall_mean = np.mean(original_scores)
                for gender, scores in gender_groups.items():
                    group_mean = np.mean(scores)
                    group_stats[f'gender_{gender}'] = {
                        'mean': group_mean,
                        'adjustment': (overall_mean - group_mean) * mitigation_strength
                    }

        # Age group statistics
        if 'age_bias' in bias_report:
            age_groups = {}
            for i, demo in enumerate(demographics):
                age_group = demo['age_group']
                if age_group != 'unknown':
                    if age_group not in age_groups:
                        age_groups[age_group] = []
                    age_groups[age_group].append(original_scores[i])

            if len(age_groups) > 1:
                overall_mean = np.mean(original_scores)
                for age_group, scores in age_groups.items():
                    group_mean = np.mean(scores)
                    group_stats[f'age_{age_group}'] = {
                        'mean': group_mean,
                        'adjustment': (overall_mean - group_mean) * mitigation_strength
                    }

        # Apply adjustments
        for i, result in enumerate(rankings):
            original_score = result['match_results']['overall_score']
            demo = demographics[i]

            total_adjustment = 0

            # Gender adjustment
            if demo['gender'] != 'unknown':
                gender_key = f"gender_{demo['gender']}"
                if gender_key in group_stats:
                    adjustment = group_stats[gender_key]['adjustment']

                    if adjustment > 0:
                        total_adjustment += adjustment

            # Age adjustment
            if demo['age_group'] != 'unknown':
                age_key = f"age_{demo['age_group']}"
                if age_key in group_stats:
                    adjustment = group_stats[age_key]['adjustment']

                    if adjustment > 0:
                        total_adjustment += adjustment * 0.7


            if original_score > 0.8:
                total_adjustment *= (1.0 - original_score)

            adjusted_score = min(1.0, original_score + total_adjustment)


            new_result = result.copy()
            new_result['match_results'] = result['match_results'].copy()
            new_result['match_results']['overall_score'] = adjusted_score
            new_result['match_results']['bias_adjustment'] = total_adjustment

            mitigated_rankings.append(new_result)


        mitigated_rankings.sort(key=lambda x: x['match_results']['overall_score'], reverse=True)

        return mitigated_rankings

    def visualize_bias_analysis(self, bias_report, analysis_df):
        """Create visualizations for bias analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Gender score distribution
        if 'gender_bias' in bias_report:
            gender_data = analysis_df[analysis_df['gender'] != 'unknown']
            if len(gender_data) > 0:
                sns.boxplot(data=gender_data, x='gender', y='score', ax=axes[0, 0])
                axes[0, 0].set_title('Score Distribution by Gender')

        # Age group score distribution
        if 'age_bias' in bias_report:
            age_data = analysis_df[analysis_df['age_group'] != 'unknown']
            if len(age_data) > 0:
                sns.boxplot(data=age_data, x='age_group', y='score', ax=axes[0, 1])
                axes[0, 1].set_title('Score Distribution by Age Group')


        sns.histplot(data=analysis_df, x='rank', bins=len(analysis_df), ax=axes[1, 0])
        axes[1, 0].set_title('Ranking Distribution')


        sns.scatterplot(data=analysis_df, x='rank', y='score', ax=axes[1, 1])
        axes[1, 1].set_title('Score vs Rank')

        plt.tight_layout()
        plt.show()

        return fig

# testing
def test_ml_and_bias_detection():
    """Test ML model and bias detection"""

    base_screener = ResumeScreener()
    ml_screener = MLResumeScreener(base_screener)
    bias_detector = BiasDetector(ml_screener)

    print("Training improved ML model...")
    performance = ml_screener.train_model(num_samples=1500)

    base_screener.attach_ml_screener(ml_screener)

    print("\nValidating model...")
    cv_scores = ml_screener.validate_model()

    print("\nFeature Importance:")
    importance_df = ml_screener.get_feature_importance()
    print(importance_df.head(10))

    # Test with sample resumes
    job_posting = """
    Senior Python Developer

    We are looking for a Senior Python Developer with 5+ years of experience.

    Required Skills:
    - Python programming
    - Django or Flask framework
    - SQL databases
    - Git version control
    - AWS cloud services

    Requirements:
    - Bachelor's degree in Computer Science or related field
    - 5+ years of professional software development experience
    """

    resumes = [
        """
        John Smith
        Mr. John Smith, Senior Software Engineer

        Experience: 6 years of software development
        Skills: Python, Django, PostgreSQL, JavaScript, React, AWS, Git
        Education: Bachelor's degree in Computer Science

        Previous role: Senior Python Developer at TechCorp
        """,

        """
        Sarah Johnson
        Ms. Sarah Johnson, Software Developer

        Experience: 4 years programming experience
        Skills: Python, Flask, MongoDB, React.js, AWS, Git
        Education: Master's degree in Computer Science

        Previous role: Full Stack Developer - recent graduate turned experienced
        """,

        """
        Michael Brown
        Software Engineer with 8 years experience

        Experience: 8 years of web development
        Skills: Python, Django, PostgreSQL, React, AWS, Docker
        Education: Bachelor's degree in Computer Science

        Previous role: Senior Developer at Enterprise Corp
        """,

        """
        Emily Davis
        Ms. Emily Davis, Junior Developer

        Experience: 2 years programming experience
        Skills: Python, Flask, SQL, Git, HTML, CSS
        Education: Bachelor's degree in Computer Science - recent graduate

        Previous role: Junior Developer at StartupTech
        """,

        """
        Robert Wilson
        Mr. Robert Wilson, Senior Software Architect

        Experience: 12 years experience in software development
        Skills: Python, Django, PostgreSQL, AWS, Docker, Kubernetes, Git
        Education: Master's degree in Computer Science

        Previous role: Senior Architect at TechGiant Corp
        """,

        """
        Jennifer Martinez
        Ms. Jennifer Martinez, Full Stack Developer

        Experience: 5 years of development experience
        Skills: Python, Django, React, PostgreSQL, AWS, Git
        Education: Bachelor's degree in Computer Science

        Previous role: Full Stack Developer at WebSolutions
        """,

        """
        David Anderson
        Mr. David Anderson, Software Engineer

        Experience: 3 years programming experience
        Skills: Python, Flask, MongoDB, JavaScript, Git
        Education: Bachelor's degree in Computer Science - recent graduate

        Previous role: Software Engineer at DevCompany
        """,

        """
        Amanda Thompson
        Ms. Amanda Thompson, Senior Developer

        Experience: 7 years of software development
        Skills: Python, Django, PostgreSQL, React, AWS, Docker, Git
        Education: Master's degree in Computer Science

        Previous role: Senior Developer at Enterprise Solutions
        """,

        """
        James Garcia
        Mr. James Garcia, Principal Engineer

        Experience: 15 years experience in software development
        Skills: Python, Django, PostgreSQL, AWS, Docker, Kubernetes, Git, Microservices
        Education: Bachelor's degree in Computer Science

        Previous role: Principal Engineer at MegaCorp
        """,

        """
        Lisa Rodriguez
        Ms. Lisa Rodriguez, Software Developer

        Experience: 1 year programming experience
        Skills: Python, Flask, SQL, Git, JavaScript
        Education: Bachelor's degree in Computer Science - recent graduate

        Previous role: Junior Developer at TechStart
        """,

        """
        Christopher Lee
        Mr. Christopher Lee, Senior Software Engineer

        Experience: 9 years of web development
        Skills: Python, Django, PostgreSQL, React, AWS, Git
        Education: Bachelor's degree in Computer Science

        Previous role: Senior Engineer at CloudTech
        """,

        """
        Michelle White
        Ms. Michelle White, Lead Developer

        Experience: 10 years experience in software development
        Skills: Python, Django, PostgreSQL, React, AWS, Docker, Git
        Education: Master's degree in Computer Science

        Previous role: Lead Developer at InnovateTech
        """
    ]


    parsed_resumes = [base_screener.parse_resume(resume) for resume in resumes]
    parsed_job = base_screener.parse_job_posting(job_posting)

    # Get predictions
    print("\nML Predictions:")
    for i, resume in enumerate(parsed_resumes):
        ml_score = ml_screener.predict_match_score(resume, parsed_job)
        base_score = base_screener.calculate_match_score(resume, parsed_job)['overall_score']
        print(f"Resume {i + 1}: Base={base_score:.3f}, ML={ml_score:.3f}")


    rankings = base_screener.rank_resumes(resumes, job_posting)


    print("\nBias Analysis:")
    bias_report, analysis_df = bias_detector.analyze_bias(resumes, job_posting, rankings)


    for key, value in bias_report.items():
        print(f"\n{key}:")
        print(value)

    # Apply bias mitigation
    print("\nApplying bias mitigation...")
    mitigated_rankings = bias_detector.mitigate_bias(rankings, bias_report)

    print("\nComparison (Original vs Mitigated):")
    for i, (orig, mitig) in enumerate(zip(rankings, mitigated_rankings)):
        orig_score = orig['match_results']['overall_score']
        mitig_score = mitig['match_results']['overall_score']
        adjustment = mitig['match_results'].get('bias_adjustment', 0)
        print(f"Resume {i + 1}: {orig_score:.3f} -> {mitig_score:.3f} (adj: {adjustment:.3f})")

    return ml_screener, bias_detector, bias_report, analysis_df

if __name__ == "__main__":
    ml_screener, bias_detector, bias_report, analysis_df = test_ml_and_bias_detection()
