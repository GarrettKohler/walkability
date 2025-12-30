"""
Comprehensive tests for walkability_utils module.
"""
import pytest
from walkability_utils import (
    normalize_walkability_score,
    categorize_walkability,
    calculate_walkability_premium,
    aggregate_neighborhood_scores,
    calculate_price_per_walkability_point,
    find_best_value_neighborhoods,
    WalkabilityScoreError
)


class TestNormalizeWalkabilityScore:
    """Tests for normalize_walkability_score function."""

    def test_score_within_range(self):
        """Test that scores within 0-100 remain unchanged."""
        assert normalize_walkability_score(50.0) == 50.0
        assert normalize_walkability_score(0.0) == 0.0
        assert normalize_walkability_score(100.0) == 100.0

    def test_score_above_100_clamped(self):
        """Test that scores above 100 are clamped to 100."""
        assert normalize_walkability_score(150.0) == 100.0
        assert normalize_walkability_score(200.5) == 100.0
        assert normalize_walkability_score(100.1) == 100.0

    def test_negative_score_raises_error(self):
        """Test that negative scores raise WalkabilityScoreError."""
        with pytest.raises(WalkabilityScoreError, match="cannot be negative"):
            normalize_walkability_score(-1.0)

        with pytest.raises(WalkabilityScoreError):
            normalize_walkability_score(-50.0)

    def test_decimal_scores(self):
        """Test handling of decimal scores."""
        assert normalize_walkability_score(45.7) == 45.7
        assert normalize_walkability_score(99.99) == 99.99
        assert normalize_walkability_score(0.01) == 0.01


class TestCategorizeWalkability:
    """Tests for categorize_walkability function."""

    def test_car_dependent_category(self):
        """Test Car-Dependent category (0-24)."""
        assert categorize_walkability(0.0) == "Car-Dependent"
        assert categorize_walkability(10.0) == "Car-Dependent"
        assert categorize_walkability(24.9) == "Car-Dependent"

    def test_somewhat_walkable_category(self):
        """Test Somewhat Walkable category (25-49)."""
        assert categorize_walkability(25.0) == "Somewhat Walkable"
        assert categorize_walkability(35.0) == "Somewhat Walkable"
        assert categorize_walkability(49.9) == "Somewhat Walkable"

    def test_very_walkable_category(self):
        """Test Very Walkable category (50-74)."""
        assert categorize_walkability(50.0) == "Very Walkable"
        assert categorize_walkability(60.0) == "Very Walkable"
        assert categorize_walkability(74.9) == "Very Walkable"

    def test_walkers_paradise_category(self):
        """Test Walker's Paradise category (75-100)."""
        assert categorize_walkability(75.0) == "Walker's Paradise"
        assert categorize_walkability(85.0) == "Walker's Paradise"
        assert categorize_walkability(100.0) == "Walker's Paradise"

    def test_invalid_score_below_range(self):
        """Test that scores below 0 raise error."""
        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            categorize_walkability(-1.0)

    def test_invalid_score_above_range(self):
        """Test that scores above 100 raise error."""
        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            categorize_walkability(101.0)

    def test_boundary_values(self):
        """Test boundary values between categories."""
        assert categorize_walkability(24.99) == "Car-Dependent"
        assert categorize_walkability(25.0) == "Somewhat Walkable"
        assert categorize_walkability(49.99) == "Somewhat Walkable"
        assert categorize_walkability(50.0) == "Very Walkable"
        assert categorize_walkability(74.99) == "Very Walkable"
        assert categorize_walkability(75.0) == "Walker's Paradise"


class TestCalculateWalkabilityPremium:
    """Tests for calculate_walkability_premium function."""

    def test_positive_premium(self):
        """Test calculation of positive premium."""
        # 50% premium
        assert calculate_walkability_premium(300000, 200000) == 50.0
        # 10% premium
        assert calculate_walkability_premium(110000, 100000) == 10.0

    def test_negative_premium(self):
        """Test calculation when walkable area is cheaper (negative premium)."""
        # -20% premium (20% discount)
        assert calculate_walkability_premium(80000, 100000) == -20.0

    def test_zero_premium(self):
        """Test when prices are equal."""
        assert calculate_walkability_premium(100000, 100000) == 0.0

    def test_rounding(self):
        """Test that result is rounded to 2 decimal places."""
        result = calculate_walkability_premium(100000, 333333)
        assert result == -70.0

        result = calculate_walkability_premium(123456, 100000)
        assert result == 23.46

    def test_negative_walkable_price_raises_error(self):
        """Test that negative walkable price raises error."""
        with pytest.raises(WalkabilityScoreError, match="cannot be negative"):
            calculate_walkability_premium(-100000, 200000)

    def test_negative_non_walkable_price_raises_error(self):
        """Test that negative non-walkable price raises error."""
        with pytest.raises(WalkabilityScoreError, match="cannot be negative"):
            calculate_walkability_premium(200000, -100000)

    def test_zero_non_walkable_price_raises_error(self):
        """Test that zero non-walkable price raises error."""
        with pytest.raises(WalkabilityScoreError, match="cannot be zero"):
            calculate_walkability_premium(200000, 0)

    def test_zero_walkable_price_allowed(self):
        """Test that zero walkable price is allowed (would be -100% premium)."""
        assert calculate_walkability_premium(0, 100000) == -100.0


class TestAggregateNeighborhoodScores:
    """Tests for aggregate_neighborhood_scores function."""

    def test_basic_aggregation(self):
        """Test basic aggregation without weights."""
        scores = [60.0, 70.0, 80.0, 90.0]
        result = aggregate_neighborhood_scores(scores)

        assert result['mean'] == 75.0
        assert result['median'] == 75.0
        assert result['min'] == 60.0
        assert result['max'] == 90.0
        assert 'weighted_mean' not in result

    def test_single_score(self):
        """Test aggregation with single score."""
        scores = [75.0]
        result = aggregate_neighborhood_scores(scores)

        assert result['mean'] == 75.0
        assert result['median'] == 75.0
        assert result['min'] == 75.0
        assert result['max'] == 75.0

    def test_weighted_aggregation(self):
        """Test aggregation with weights."""
        scores = [60.0, 80.0, 100.0]
        weights = [0.2, 0.3, 0.5]  # Sum = 1.0

        result = aggregate_neighborhood_scores(scores, weights)

        # Weighted mean = 60*0.2 + 80*0.3 + 100*0.5 = 12 + 24 + 50 = 86
        assert result['weighted_mean'] == 86.0
        assert result['mean'] == 80.0

    def test_odd_number_of_scores_median(self):
        """Test median calculation with odd number of scores."""
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = aggregate_neighborhood_scores(scores)
        assert result['median'] == 30.0

    def test_empty_scores_raises_error(self):
        """Test that empty scores list raises error."""
        with pytest.raises(WalkabilityScoreError, match="cannot be empty"):
            aggregate_neighborhood_scores([])

    def test_invalid_score_raises_error(self):
        """Test that invalid scores raise error."""
        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            aggregate_neighborhood_scores([50.0, 101.0, 75.0])

        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            aggregate_neighborhood_scores([50.0, -1.0, 75.0])

    def test_weights_length_mismatch_raises_error(self):
        """Test that mismatched weights and scores length raises error."""
        with pytest.raises(WalkabilityScoreError, match="same length"):
            aggregate_neighborhood_scores([60.0, 80.0], [0.5, 0.3, 0.2])

    def test_weights_not_summing_to_one_raises_error(self):
        """Test that weights not summing to 1.0 raise error."""
        with pytest.raises(WalkabilityScoreError, match="must sum to 1.0"):
            aggregate_neighborhood_scores([60.0, 80.0], [0.3, 0.3])

    def test_negative_weights_raise_error(self):
        """Test that negative weights raise error."""
        with pytest.raises(WalkabilityScoreError, match="cannot be negative"):
            aggregate_neighborhood_scores([60.0, 80.0], [-0.2, 1.2])

    def test_weights_with_floating_point_tolerance(self):
        """Test that small floating point errors in weights are tolerated."""
        scores = [60.0, 80.0, 100.0]
        weights = [0.2, 0.3, 0.500001]  # Sum slightly over 1.0 but within tolerance

        result = aggregate_neighborhood_scores(scores, weights)
        assert 'weighted_mean' in result


class TestCalculatePricePerWalkabilityPoint:
    """Tests for calculate_price_per_walkability_point function."""

    def test_basic_calculation(self):
        """Test basic price per point calculation."""
        # $300,000 / 75 points = $4,000 per point
        assert calculate_price_per_walkability_point(300000, 75.0) == 4000.0

    def test_low_walkability_high_price_per_point(self):
        """Test that low walkability yields high price per point."""
        # $200,000 / 20 points = $10,000 per point
        assert calculate_price_per_walkability_point(200000, 20.0) == 10000.0

    def test_high_walkability_low_price_per_point(self):
        """Test that high walkability yields low price per point."""
        # $300,000 / 100 points = $3,000 per point
        assert calculate_price_per_walkability_point(300000, 100.0) == 3000.0

    def test_rounding(self):
        """Test that result is rounded to 2 decimal places."""
        # $100,000 / 33 points = $3,030.30...
        result = calculate_price_per_walkability_point(100000, 33.0)
        assert result == 3030.30

    def test_negative_price_raises_error(self):
        """Test that negative price raises error."""
        with pytest.raises(WalkabilityScoreError, match="cannot be negative"):
            calculate_price_per_walkability_point(-100000, 50.0)

    def test_zero_walkability_raises_error(self):
        """Test that zero walkability score raises error."""
        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            calculate_price_per_walkability_point(100000, 0)

    def test_negative_walkability_raises_error(self):
        """Test that negative walkability score raises error."""
        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            calculate_price_per_walkability_point(100000, -50.0)

    def test_walkability_above_100_raises_error(self):
        """Test that walkability score above 100 raises error."""
        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            calculate_price_per_walkability_point(100000, 101.0)

    def test_zero_price_allowed(self):
        """Test that zero price is allowed."""
        assert calculate_price_per_walkability_point(0, 50.0) == 0.0


class TestFindBestValueNeighborhoods:
    """Tests for find_best_value_neighborhoods function."""

    def test_basic_sorting(self):
        """Test that neighborhoods are sorted by value (walkability/price ratio)."""
        neighborhoods = [
            {'name': 'Downtown', 'price': 500000, 'walkability_score': 90},  # value: 0.18
            {'name': 'Suburbs', 'price': 300000, 'walkability_score': 30},   # value: 0.10
            {'name': 'Midtown', 'price': 400000, 'walkability_score': 80},   # value: 0.20
        ]

        result = find_best_value_neighborhoods(neighborhoods, top_n=3)

        assert len(result) == 3
        assert result[0]['name'] == 'Midtown'  # Best value
        assert result[1]['name'] == 'Downtown'
        assert result[2]['name'] == 'Suburbs'  # Worst value

    def test_top_n_limiting(self):
        """Test that only top N neighborhoods are returned."""
        neighborhoods = [
            {'name': 'A', 'price': 100000, 'walkability_score': 50},
            {'name': 'B', 'price': 100000, 'walkability_score': 60},
            {'name': 'C', 'price': 100000, 'walkability_score': 70},
            {'name': 'D', 'price': 100000, 'walkability_score': 80},
        ]

        result = find_best_value_neighborhoods(neighborhoods, top_n=2)

        assert len(result) == 2
        assert result[0]['name'] == 'D'  # Highest walkability
        assert result[1]['name'] == 'C'

    def test_value_score_calculation(self):
        """Test that value_score is correctly calculated and added."""
        neighborhoods = [
            {'name': 'Test', 'price': 100000, 'walkability_score': 80},
        ]

        result = find_best_value_neighborhoods(neighborhoods, top_n=1)

        assert 'value_score' in result[0]
        # value_score = 80 / 100000 * 1000 = 0.8
        assert result[0]['value_score'] == 0.8

    def test_same_price_different_walkability(self):
        """Test sorting when prices are equal."""
        neighborhoods = [
            {'name': 'Low', 'price': 200000, 'walkability_score': 40},
            {'name': 'High', 'price': 200000, 'walkability_score': 80},
            {'name': 'Medium', 'price': 200000, 'walkability_score': 60},
        ]

        result = find_best_value_neighborhoods(neighborhoods, top_n=3)

        assert result[0]['name'] == 'High'
        assert result[1]['name'] == 'Medium'
        assert result[2]['name'] == 'Low'

    def test_original_data_preserved(self):
        """Test that original neighborhood data is preserved."""
        neighborhoods = [
            {'name': 'Test', 'price': 300000, 'walkability_score': 75},
        ]

        result = find_best_value_neighborhoods(neighborhoods, top_n=1)

        assert result[0]['name'] == 'Test'
        assert result[0]['price'] == 300000
        assert result[0]['walkability_score'] == 75

    def test_empty_neighborhoods_raises_error(self):
        """Test that empty neighborhoods list raises error."""
        with pytest.raises(WalkabilityScoreError, match="cannot be empty"):
            find_best_value_neighborhoods([])

    def test_invalid_top_n_raises_error(self):
        """Test that invalid top_n raises error."""
        neighborhoods = [
            {'name': 'Test', 'price': 100000, 'walkability_score': 50},
        ]

        with pytest.raises(WalkabilityScoreError, match="must be at least 1"):
            find_best_value_neighborhoods(neighborhoods, top_n=0)

        with pytest.raises(WalkabilityScoreError, match="must be at least 1"):
            find_best_value_neighborhoods(neighborhoods, top_n=-1)

    def test_missing_required_field_raises_error(self):
        """Test that missing required fields raise error."""
        neighborhoods = [
            {'name': 'Test', 'price': 100000},  # Missing walkability_score
        ]

        with pytest.raises(WalkabilityScoreError, match="must have"):
            find_best_value_neighborhoods(neighborhoods)

    def test_invalid_price_raises_error(self):
        """Test that invalid prices raise error."""
        neighborhoods = [
            {'name': 'Test', 'price': 0, 'walkability_score': 50},
        ]

        with pytest.raises(WalkabilityScoreError, match="must be positive"):
            find_best_value_neighborhoods(neighborhoods)

        neighborhoods = [
            {'name': 'Test', 'price': -100000, 'walkability_score': 50},
        ]

        with pytest.raises(WalkabilityScoreError, match="must be positive"):
            find_best_value_neighborhoods(neighborhoods)

    def test_invalid_walkability_score_raises_error(self):
        """Test that invalid walkability scores raise error."""
        neighborhoods = [
            {'name': 'Test', 'price': 100000, 'walkability_score': 0},
        ]

        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            find_best_value_neighborhoods(neighborhoods)

        neighborhoods = [
            {'name': 'Test', 'price': 100000, 'walkability_score': 101},
        ]

        with pytest.raises(WalkabilityScoreError, match="must be between 0 and 100"):
            find_best_value_neighborhoods(neighborhoods)

    def test_top_n_greater_than_list_length(self):
        """Test behavior when top_n exceeds list length."""
        neighborhoods = [
            {'name': 'A', 'price': 100000, 'walkability_score': 50},
            {'name': 'B', 'price': 100000, 'walkability_score': 60},
        ]

        result = find_best_value_neighborhoods(neighborhoods, top_n=10)

        # Should return all available neighborhoods
        assert len(result) == 2

    def test_complex_scenario(self):
        """Test a complex real-world scenario."""
        neighborhoods = [
            {'name': 'Urban Core', 'price': 800000, 'walkability_score': 95},
            {'name': 'Trendy District', 'price': 600000, 'walkability_score': 85},
            {'name': 'Family Suburb', 'price': 400000, 'walkability_score': 45},
            {'name': 'Rural Area', 'price': 200000, 'walkability_score': 15},
            {'name': 'Transit Village', 'price': 500000, 'walkability_score': 88},
        ]

        result = find_best_value_neighborhoods(neighborhoods, top_n=3)

        # Transit Village should be best value (88/500000 = 0.176)
        # Trendy District second (85/600000 = 0.142)
        # Urban Core third (95/800000 = 0.119)
        assert result[0]['name'] == 'Transit Village'
        assert result[1]['name'] == 'Trendy District'
        assert result[2]['name'] == 'Urban Core'
