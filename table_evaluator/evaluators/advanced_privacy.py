"""Advanced privacy evaluation functionality."""

import pandas as pd
from loguru import logger

from table_evaluator.metrics.privacy import (
    calculate_k_anonymity,
    identify_quasi_identifiers,
    simulate_membership_inference_attack,
)


class AdvancedPrivacyEvaluator:
    """Advanced privacy evaluation using modern privacy analysis techniques."""

    def __init__(self, verbose: bool = False):
        """
        Initialize the advanced privacy evaluator.

        Args:
            verbose: Whether to print detailed output during evaluation
        """
        self.verbose = verbose

    def k_anonymity_evaluation(
        self,
        synthetic_data: pd.DataFrame,
        quasi_identifiers: list[str] | None = None,
        sensitive_attributes: list[str] | None = None,
        auto_detect_qi: bool = True,
    ) -> dict:
        """
        Comprehensive k-anonymity evaluation of synthetic data.

        Args:
            synthetic_data: DataFrame containing synthetic data
            quasi_identifiers: List of quasi-identifier column names
            sensitive_attributes: List of sensitive attribute column names
            auto_detect_qi: Whether to automatically detect quasi-identifiers

        Returns:
            Dictionary containing k-anonymity analysis results
        """
        if self.verbose:
            print('Performing k-anonymity analysis...')

        results = {'analysis': {}, 'recommendations': [], 'privacy_score': 0.0}

        try:
            # Auto-detect quasi-identifiers if not provided
            if quasi_identifiers is None and auto_detect_qi:
                quasi_identifiers = identify_quasi_identifiers(synthetic_data)
                if self.verbose:
                    print(f'Auto-detected quasi-identifiers: {quasi_identifiers}')

            if not quasi_identifiers:
                results['analysis'] = {
                    'error': 'No quasi-identifiers specified or detected',
                    'k_value': float('inf'),
                    'anonymity_level': 'Perfect (no quasi-identifiers)',
                }
                results['privacy_score'] = 1.0
                return results

            # Calculate k-anonymity metrics
            k_anon_results = calculate_k_anonymity(synthetic_data, quasi_identifiers, sensitive_attributes)

            results['analysis'] = k_anon_results
            results['quasi_identifiers_used'] = quasi_identifiers
            if sensitive_attributes:
                results['sensitive_attributes_used'] = sensitive_attributes

            # Generate specific recommendations
            k_value = k_anon_results.get('k_value', 0)
            violations = k_anon_results.get('violations', 0)

            if k_value < 2:
                results['recommendations'].append(
                    'Critical: Some records are uniquely identifiable. Apply generalization or suppression techniques.'
                )
            elif k_value < 3:
                results['recommendations'].append(
                    'Warning: Low k-anonymity detected. Consider increasing generalization.'
                )
            elif k_value < 5:
                results['recommendations'].append(
                    'Moderate k-anonymity. Consider minor improvements for better privacy.'
                )
            else:
                results['recommendations'].append('Good k-anonymity level achieved.')

            if violations > 0:
                results['recommendations'].append(
                    f'{violations} equivalence classes violate k-anonymity. '
                    'Focus on these specific groups for improvement.'
                )

            # Calculate privacy score
            results['privacy_score'] = self._calculate_k_anonymity_score(k_value, violations)

        except Exception as e:
            logger.error(f'Error in k-anonymity evaluation: {e}')
            results['analysis'] = {'error': str(e)}
            results['recommendations'] = ['Error in analysis - check data format']

        return results

    def membership_inference_evaluation(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_columns: list[str] | None = None,
        attack_models: list[str] | None = None,
    ) -> dict:
        """
        Membership inference attack evaluation.

        Args:
            real_data: Original dataset
            synthetic_data: Synthetic dataset
            target_columns: Columns to use for attack (auto-detected if None)
            attack_models: List of attack models to use

        Returns:
            Dictionary containing membership inference attack results
        """
        if self.verbose:
            print('Performing membership inference attack simulation...')

        results = {
            'attack_results': {},
            'vulnerability_assessment': {},
            'recommendations': [],
        }

        try:
            # Simulate membership inference attacks
            attack_results = simulate_membership_inference_attack(real_data, synthetic_data, target_columns)

            results['attack_results'] = attack_results

            # Assess vulnerability
            if 'summary' in attack_results:
                summary = attack_results['summary']
                max_accuracy = summary.get('max_attack_accuracy', 0.5)
                vulnerability_level = summary.get('privacy_vulnerability', 'Unknown')

                results['vulnerability_assessment'] = {
                    'max_attack_accuracy': max_accuracy,
                    'vulnerability_level': vulnerability_level,
                    'privacy_risk_score': self._calculate_inference_risk_score(max_accuracy),
                    'baseline_accuracy': 0.5,
                }

                # Generate recommendations
                if max_accuracy > 0.75:
                    results['recommendations'].extend(
                        [
                            'High privacy risk: Attackers can distinguish real from synthetic data.',
                            'Consider differential privacy mechanisms.',
                            'Increase synthesis model complexity or training data size.',
                            'Add noise to sensitive attributes.',
                        ]
                    )
                elif max_accuracy > 0.6:
                    results['recommendations'].extend(
                        [
                            'Moderate privacy risk detected.',
                            'Consider adding privacy-preserving techniques.',
                            'Review data preprocessing and synthesis parameters.',
                        ]
                    )
                else:
                    results['recommendations'].append('Low privacy risk - good privacy protection achieved.')

            # Model-specific analysis
            model_performances = []
            for model_name, model_results in attack_results.items():
                if model_name != 'summary' and 'error' not in model_results:
                    model_performances.append(
                        {
                            'model': model_name,
                            'accuracy': model_results.get('accuracy', 0),
                            'auc_score': model_results.get('auc_score', 0),
                            'risk_level': model_results.get('privacy_risk', 'Unknown'),
                        }
                    )

            results['model_performances'] = model_performances

        except Exception as e:
            logger.error(f'Error in membership inference evaluation: {e}')
            results['attack_results'] = {'error': str(e)}
            results['recommendations'] = ['Error in attack simulation - check data compatibility']

        return results

    def comprehensive_privacy_evaluation(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        quasi_identifiers: list[str] | None = None,
        sensitive_attributes: list[str] | None = None,
        include_k_anonymity: bool = True,
        include_membership_inference: bool = True,
    ) -> dict:
        """
        Comprehensive privacy evaluation combining multiple techniques.

        Args:
            real_data: Original dataset
            synthetic_data: Synthetic dataset
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attributes: List of sensitive attribute columns
            include_k_anonymity: Whether to include k-anonymity analysis
            include_membership_inference: Whether to include membership inference analysis

        Returns:
            Dictionary with comprehensive privacy analysis results
        """
        if self.verbose:
            print('Running comprehensive privacy evaluation...')

        results = {
            'k_anonymity': {},
            'membership_inference': {},
            'overall_assessment': {},
            'combined_recommendations': [],
        }

        # K-anonymity evaluation
        if include_k_anonymity:
            try:
                results['k_anonymity'] = self.k_anonymity_evaluation(
                    synthetic_data, quasi_identifiers, sensitive_attributes
                )
            except Exception as e:
                logger.error(f'K-anonymity evaluation failed: {e}')
                results['k_anonymity'] = {'error': str(e)}

        # Membership inference evaluation
        if include_membership_inference:
            try:
                results['membership_inference'] = self.membership_inference_evaluation(real_data, synthetic_data)
            except Exception as e:
                logger.error(f'Membership inference evaluation failed: {e}')
                results['membership_inference'] = {'error': str(e)}

        # Combined analysis
        try:
            results['overall_assessment'] = self._assess_combined_privacy_risk(
                results['k_anonymity'], results['membership_inference']
            )
            results['combined_recommendations'] = self._generate_combined_recommendations(
                results['k_anonymity'], results['membership_inference']
            )
        except Exception as e:
            logger.error(f'Combined privacy assessment failed: {e}')
            results['overall_assessment'] = {'error': str(e)}

        return results

    def privacy_risk_dashboard(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs) -> dict:
        """
        Generate a privacy risk dashboard with key metrics and visualizations.

        Args:
            real_data: Original dataset
            synthetic_data: Synthetic dataset
            **kwargs: Additional parameters for evaluation

        Returns:
            Dictionary with dashboard-ready privacy metrics
        """
        if self.verbose:
            print('Generating privacy risk dashboard...')

        # Run comprehensive evaluation
        comprehensive_results = self.comprehensive_privacy_evaluation(real_data, synthetic_data, **kwargs)

        # Extract key metrics for dashboard
        dashboard = {
            'risk_summary': {},
            'key_metrics': {},
            'alerts': [],
            'recommendations': [],
        }

        try:
            # Overall risk level
            overall_assessment = comprehensive_results.get('overall_assessment', {})
            dashboard['risk_summary'] = {
                'overall_risk_level': overall_assessment.get('overall_risk_level', 'Unknown'),
                'privacy_score': overall_assessment.get('privacy_score', 0.0),
                'risk_factors': overall_assessment.get('identified_risks', []),
            }

            # Key metrics
            k_anon = comprehensive_results.get('k_anonymity', {}).get('analysis', {})
            membership = comprehensive_results.get('membership_inference', {}).get('vulnerability_assessment', {})

            dashboard['key_metrics'] = {
                'k_anonymity_value': k_anon.get('k_value', 'N/A'),
                'k_anonymity_level': k_anon.get('anonymity_level', 'N/A'),
                'max_attack_accuracy': membership.get('max_attack_accuracy', 'N/A'),
                'vulnerability_level': membership.get('vulnerability_level', 'N/A'),
            }

            # Generate alerts
            if k_anon.get('k_value', float('inf')) < 3:
                dashboard['alerts'].append('Low k-anonymity detected')

            if membership.get('max_attack_accuracy', 0) > 0.7:
                dashboard['alerts'].append('High membership inference risk')

            # Combined recommendations
            dashboard['recommendations'] = comprehensive_results.get('combined_recommendations', [])

        except Exception as e:
            logger.error(f'Error generating privacy dashboard: {e}')
            dashboard['error'] = str(e)

        return dashboard

    def _calculate_k_anonymity_score(self, k_value: int, violations: int) -> float:
        """Calculate privacy score based on k-anonymity metrics."""
        if k_value == 0:
            return 0.0

        # Base score from k-value
        base_score = min(1.0, k_value / 10.0)  # Max score at k=10

        # Penalty for violations
        violation_penalty = violations * 0.1

        return max(0.0, base_score - violation_penalty)

    def _calculate_inference_risk_score(self, attack_accuracy: float) -> float:
        """Calculate risk score from membership inference attack accuracy."""
        # Convert accuracy to risk score (0.5 = no risk, 1.0 = maximum risk)
        if attack_accuracy <= 0.5:
            return 0.0

        # Scale from 0.5-1.0 accuracy to 0.0-1.0 risk
        return (attack_accuracy - 0.5) * 2.0

    def _assess_combined_privacy_risk(self, k_anon_results: dict, membership_results: dict) -> dict:
        """Assess overall privacy risk from combined analyses."""
        risk_factors = []

        # K-anonymity risk factors
        k_anon_analysis = k_anon_results.get('analysis', {})
        k_value = k_anon_analysis.get('k_value', float('inf'))

        if k_value < 2:
            risk_factors.append('Critical k-anonymity risk')
        elif k_value < 5:
            risk_factors.append('Moderate k-anonymity risk')

        # Membership inference risk factors
        vulnerability = membership_results.get('vulnerability_assessment', {})
        max_accuracy = vulnerability.get('max_attack_accuracy', 0.5)

        if max_accuracy > 0.75:
            risk_factors.append('High membership inference risk')
        elif max_accuracy > 0.6:
            risk_factors.append('Moderate membership inference risk')

        # Overall risk level
        if any('Critical' in factor or 'High' in factor for factor in risk_factors):
            overall_risk = 'High'
        elif any('Moderate' in factor for factor in risk_factors):
            overall_risk = 'Moderate'
        else:
            overall_risk = 'Low'

        # Combined privacy score
        k_anon_score = k_anon_results.get('privacy_score', 1.0)
        inference_risk = vulnerability.get('privacy_risk_score', 0.0)
        combined_score = k_anon_score * (1.0 - inference_risk)

        return {
            'overall_risk_level': overall_risk,
            'identified_risks': risk_factors,
            'privacy_score': max(0.0, combined_score),
            'risk_breakdown': {
                'k_anonymity_score': k_anon_score,
                'inference_risk_score': inference_risk,
            },
        }

    def _generate_combined_recommendations(self, k_anon_results: dict, membership_results: dict) -> list[str]:
        """Generate combined recommendations from all privacy analyses."""
        recommendations = []

        # Add k-anonymity recommendations
        k_anon_recs = k_anon_results.get('recommendations', [])
        recommendations.extend(k_anon_recs)

        # Add membership inference recommendations
        membership_recs = membership_results.get('recommendations', [])
        recommendations.extend(membership_recs)

        # Add combined recommendations
        k_value = k_anon_results.get('analysis', {}).get('k_value', float('inf'))
        max_accuracy = membership_results.get('vulnerability_assessment', {}).get('max_attack_accuracy', 0.5)

        if k_value < 3 and max_accuracy > 0.7:
            recommendations.append(
                'Both k-anonymity and membership inference risks detected. '
                'Consider comprehensive privacy-preserving data synthesis methods.'
            )

        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)

        return unique_recommendations
