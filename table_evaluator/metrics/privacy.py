"""Privacy attack simulations and privacy metrics implementation."""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from table_evaluator.constants import RANDOM_SEED
from table_evaluator.models.privacy_models import (
    AttackModelResult,
    KAnonymityAnalysis,
    LDiversityAnalysis,
    MembershipInferenceAnalysis,
    StatisticalDistribution,
)


def _convert_statistical_distribution(describe_dict: dict) -> StatisticalDistribution:
    """Convert pandas describe() output to StatisticalDistribution model."""
    return StatisticalDistribution(
        count=describe_dict.get('count', 0.0),
        mean=describe_dict.get('mean', 0.0),
        std=describe_dict.get('std', 0.0),
        min=describe_dict.get('min', 0.0),
        percentile_25=describe_dict.get('25%', 0.0),
        percentile_50=describe_dict.get('50%', 0.0),
        percentile_75=describe_dict.get('75%', 0.0),
        max=describe_dict.get('max', 0.0),
    )


def identify_quasi_identifiers(df: pd.DataFrame, max_unique_ratio: float = 0.9, min_unique_count: int = 2) -> list[str]:
    """
    Automatically identify potential quasi-identifiers in a dataset.

    Quasi-identifiers are attributes that can be used to re-identify individuals
    when combined with external information.

    Args:
        df: DataFrame to analyze
        max_unique_ratio: Maximum ratio of unique values to total records
        min_unique_count: Minimum number of unique values required

    Returns:
        List of column names identified as potential quasi-identifiers
    """
    quasi_identifiers = []

    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df)
        unique_ratio = unique_count / total_count

        # Identify columns with moderate uniqueness (not completely unique, not constant)
        if (
            min_unique_count <= unique_count and unique_ratio <= max_unique_ratio and unique_ratio > 0.01
        ):  # Not nearly constant
            quasi_identifiers.append(col)

    return quasi_identifiers


def calculate_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_attributes: list[str] | None = None,
) -> KAnonymityAnalysis:
    """
    Calculate k-anonymity metrics for a dataset.

    k-anonymity ensures that each record is indistinguishable from at least
    k-1 other records with respect to quasi-identifiers.

    Args:
        df: DataFrame to analyze
        quasi_identifiers: List of quasi-identifier column names
        sensitive_attributes: List of sensitive attribute column names

    Returns:
        KAnonymityAnalysis: K-anonymity analysis results
    """
    if not quasi_identifiers:
        return KAnonymityAnalysis(
            k_value=0,
            anonymity_level='Error',
            violations=0,
            equivalence_classes=0,
            avg_class_size=0.0,
            class_size_distribution=StatisticalDistribution(
                count=0.0, mean=0.0, std=0.0, min=0.0, percentile_25=0.0, percentile_50=0.0, percentile_75=0.0, max=0.0
            ),
            sample_size=len(df) if df is not None else 0,
        )

    # Group by quasi-identifiers
    try:
        grouped = df.groupby(quasi_identifiers).size().reset_index(name='count')

        # Find minimum group size (k-value)
        k_value = grouped['count'].min()

        # Count violations (groups smaller than desired k)
        violations = (grouped['count'] < k_value).sum()

        # Equivalence class statistics
        n_equivalence_classes = len(grouped)
        avg_class_size = grouped['count'].mean()

        # Anonymity level assessment
        if k_value >= 5:
            anonymity_level = 'Good'
        elif k_value >= 3:
            anonymity_level = 'Moderate'
        elif k_value >= 2:
            anonymity_level = 'Weak'
        else:
            anonymity_level = 'Poor'

        # Convert class size distribution
        class_size_distribution = _convert_statistical_distribution(grouped['count'].describe().to_dict())

        # Handle l-diversity results
        l_diversity_results = {}
        if sensitive_attributes:
            for sensitive_attr in sensitive_attributes:
                try:
                    # Group by quasi-identifiers and analyze diversity in sensitive attribute
                    diversity_analysis = (
                        df.groupby(quasi_identifiers)[sensitive_attr]
                        .apply(lambda x: x.nunique())
                        .reset_index(name='l_value')
                    )

                    min_l = diversity_analysis['l_value'].min()
                    avg_l = diversity_analysis['l_value'].mean()

                    # Check for violations (classes with l < 2)
                    l_violations = (diversity_analysis['l_value'] < 2).sum()

                    l_diversity_results[sensitive_attr] = LDiversityAnalysis(
                        min_l_value=int(min_l),
                        avg_l_value=float(avg_l),
                        l_violations=int(l_violations),
                        diversity_distribution=_convert_statistical_distribution(
                            diversity_analysis['l_value'].describe().to_dict()
                        ),
                    )

                except Exception as e:
                    logger.error(f'Error analyzing l-diversity for {sensitive_attr}: {e}')
                    # Skip this sensitive attribute if it fails

        return KAnonymityAnalysis(
            k_value=int(k_value),
            anonymity_level=anonymity_level,
            violations=int(violations),
            equivalence_classes=int(n_equivalence_classes),
            avg_class_size=float(avg_class_size),
            class_size_distribution=class_size_distribution,
            l_diversity_results=l_diversity_results,
            sample_size=len(df),
        )

    except Exception as e:
        logger.error(f'Error calculating k-anonymity: {e}')
        return KAnonymityAnalysis(
            k_value=0,
            anonymity_level='Error',
            violations=0,
            equivalence_classes=0,
            avg_class_size=0.0,
            class_size_distribution=StatisticalDistribution(
                count=0.0, mean=0.0, std=0.0, min=0.0, percentile_25=0.0, percentile_50=0.0, percentile_75=0.0, max=0.0
            ),
            sample_size=0,
        )


def simulate_membership_inference_attack(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_columns: list[str] | None = None,
    test_size: float = 0.3,
    random_state: int = RANDOM_SEED,
) -> MembershipInferenceAnalysis:
    """
    Simulate membership inference attacks to assess privacy risks.

    A membership inference attack tries to determine whether a specific
    record was part of the training data used to generate synthetic data.

    Args:
        real_data: Original dataset
        synthetic_data: Synthetic dataset
        target_columns: Columns to use for attack (if None, use all)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        MembershipInferenceAnalysis: Attack simulation results
    """
    if target_columns is None:
        # Use numerical and low-cardinality categorical columns
        target_columns = []
        for col in real_data.columns:
            if real_data[col].dtype in ['int64', 'float64'] or real_data[col].nunique() <= 20:
                target_columns.append(col)

    if not target_columns:
        return MembershipInferenceAnalysis(
            max_attack_accuracy=0.5,
            avg_attack_accuracy=0.5,
            privacy_vulnerability='Low',
            recommendation='No suitable columns found for attack simulation',
            sample_size=len(real_data),
        )

    try:
        # Prepare attack dataset
        # Sample equal amounts from real and synthetic data
        min_samples = min(len(real_data), len(synthetic_data))
        sample_size = min(min_samples, 5000)  # Limit for computational efficiency

        real_sample = real_data.sample(n=sample_size, random_state=random_state)
        synthetic_sample = synthetic_data.sample(n=sample_size, random_state=random_state)

        # Create attack dataset: real=1, synthetic=0
        real_features = real_sample[target_columns].copy()
        synthetic_features = synthetic_sample[target_columns].copy()

        # Encode categorical variables
        encoded_real = encode_features(real_features)
        encoded_synthetic = encode_features(synthetic_features)

        # Combine datasets
        X = pd.concat([encoded_real, encoded_synthetic], ignore_index=True)
        y = np.concatenate([np.ones(len(encoded_real)), np.zeros(len(encoded_synthetic))])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train multiple attack models
        models = {
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
        }

        model_results = {}
        accuracies = []

        for model_name, model in models.items():
            try:
                # Train attack model
                if model_name == 'logistic_regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Calculate attack metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)

                # Assess privacy risk
                if accuracy > 0.75:
                    risk_level = 'High'
                elif accuracy > 0.6:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'

                model_results[model_name] = AttackModelResult(
                    accuracy=float(accuracy),
                    precision=float(precision),
                    recall=float(recall),
                    auc_score=float(auc_score),
                    privacy_risk=risk_level,
                    baseline_accuracy=0.5,
                )

                accuracies.append(accuracy)

            except Exception:  # noqa: PERF203
                logger.exception(f'Error training {model_name}')
                # Skip this model if it fails

        # Overall assessment
        if accuracies:
            max_accuracy = max(accuracies)
            avg_accuracy = np.mean(accuracies)

            # Determine vulnerability level
            if max_accuracy > 0.75:
                vulnerability_level = 'High'
            elif max_accuracy > 0.6:
                vulnerability_level = 'Medium'
            else:
                vulnerability_level = 'Low'

            # Generate recommendation
            recommendation = generate_privacy_recommendation(max_accuracy)

            return MembershipInferenceAnalysis(
                logistic_regression=model_results.get('logistic_regression'),
                random_forest=model_results.get('random_forest'),
                max_attack_accuracy=float(max_accuracy),
                avg_attack_accuracy=float(avg_accuracy),
                privacy_vulnerability=vulnerability_level,
                recommendation=recommendation,
                sample_size=min_samples,
            )
        return MembershipInferenceAnalysis(
            max_attack_accuracy=0.5,
            avg_attack_accuracy=0.5,
            privacy_vulnerability='Low',
            recommendation='No attack models could be trained successfully',
            sample_size=min(len(real_data), len(synthetic_data)),
        )

    except Exception as e:
        logger.exception('Error in membership inference attack simulation')
        return MembershipInferenceAnalysis(
            max_attack_accuracy=0.5,
            avg_attack_accuracy=0.5,
            privacy_vulnerability='Low',
            recommendation=f'Error in attack simulation: {e}',
            sample_size=min(len(real_data), len(synthetic_data)),
        )


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for machine learning models."""
    encoded_df = df.copy()

    for col in encoded_df.columns:
        if encoded_df[col].dtype == 'object':
            # Use label encoding for categorical variables
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))

    return encoded_df


def generate_privacy_recommendation(attack_accuracy: float) -> str:
    """Generate privacy improvement recommendations based on attack success."""
    if attack_accuracy > 0.75:
        return (
            'High privacy risk detected. Consider adding differential privacy, '
            'increasing data synthesis complexity, or reducing data resolution.'
        )
    if attack_accuracy > 0.6:
        return (
            'Moderate privacy risk. Consider adding noise to sensitive attributes '
            'or using more sophisticated synthesis methods.'
        )
    return 'Low privacy risk. Current synthesis method provides good privacy protection.'


def comprehensive_privacy_analysis(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    quasi_identifiers: list[str] | None = None,
    sensitive_attributes: list[str] | None = None,
) -> dict:
    """
    Comprehensive privacy analysis combining k-anonymity and membership inference.

    Args:
        real_data: Original dataset
        synthetic_data: Synthetic dataset
        quasi_identifiers: List of quasi-identifier columns (auto-detected if None)
        sensitive_attributes: List of sensitive attribute columns

    Returns:
        Dictionary with comprehensive privacy analysis results
    """
    results = {'k_anonymity': None, 'membership_inference': None, 'overall_assessment': {}}

    # Auto-detect quasi-identifiers if not provided
    if quasi_identifiers is None:
        quasi_identifiers = identify_quasi_identifiers(synthetic_data)

    # K-anonymity analysis
    try:
        k_anonymity_result = calculate_k_anonymity(synthetic_data, quasi_identifiers, sensitive_attributes)
        results['k_anonymity'] = k_anonymity_result.model_dump()
    except Exception as e:
        logger.error(f'K-anonymity analysis failed: {e}')
        results['k_anonymity'] = {'error': str(e)}

    # Membership inference attack simulation
    try:
        membership_result = simulate_membership_inference_attack(real_data, synthetic_data)
        results['membership_inference'] = membership_result.model_dump()
    except Exception as e:
        logger.error(f'Membership inference analysis failed: {e}')
        results['membership_inference'] = {'error': str(e)}

    # Overall privacy assessment
    try:
        results['overall_assessment'] = assess_overall_privacy_risk(results)
    except Exception as e:
        logger.error(f'Overall assessment failed: {e}')
        results['overall_assessment'] = {'error': str(e)}

    return results


def assess_overall_privacy_risk(privacy_results: dict) -> dict:
    """Assess overall privacy risk based on multiple analyses."""
    k_anon = privacy_results.get('k_anonymity', {})
    membership = privacy_results.get('membership_inference', {})

    risks = []

    # K-anonymity risk
    k_value = k_anon.get('k_value', 0)
    if k_value < 2:
        risks.append('High k-anonymity risk')
    elif k_value < 5:
        risks.append('Moderate k-anonymity risk')

    # Membership inference risk
    max_accuracy = membership.get('max_attack_accuracy', 0.5)
    if max_accuracy > 0.75:
        risks.append('High membership inference risk')
    elif max_accuracy > 0.6:
        risks.append('Moderate membership inference risk')

    # Overall risk level
    if any('High' in risk for risk in risks):
        overall_risk = 'High'
    elif any('Moderate' in risk for risk in risks):
        overall_risk = 'Moderate'
    else:
        overall_risk = 'Low'

    return {
        'overall_risk_level': overall_risk,
        'identified_risks': risks,
        'privacy_score': calculate_privacy_score(k_anon, membership),
        'recommendations': generate_comprehensive_recommendations(risks),
    }


def calculate_privacy_score(k_anon: dict, membership: dict) -> float:
    """Calculate overall privacy score (0-1, higher is better)."""
    score = 1.0

    # Penalize based on k-anonymity
    k_value = k_anon.get('k_value', 1)
    if k_value < 5:
        score *= k_value / 5.0

    # Penalize based on membership inference
    max_accuracy = membership.get('max_attack_accuracy', 0.5)
    # Convert accuracy to privacy score (0.5 = perfect, 1.0 = worst)
    privacy_factor = max(0, 1.0 - (max_accuracy - 0.5) * 2)
    score *= privacy_factor

    return max(0.0, min(1.0, score))


def generate_comprehensive_recommendations(risks: list[str]) -> list[str]:
    """Generate comprehensive privacy improvement recommendations."""
    recommendations = []

    if any('k-anonymity' in risk for risk in risks):
        recommendations.append('Improve k-anonymity by generalizing or suppressing quasi-identifiers')

    if any('membership inference' in risk for risk in risks):
        recommendations.append('Add differential privacy or increase synthesis model complexity')

    if not risks:
        recommendations.append('Good privacy protection achieved. Continue monitoring with larger datasets.')

    return recommendations
