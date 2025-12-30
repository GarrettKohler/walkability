"""
Walkability utility functions for analyzing EPA Walkability Scores and housing data.
"""
from typing import List, Dict, Optional, Tuple
from statistics import mean, median


class WalkabilityScoreError(Exception):
    """Custom exception for walkability score errors."""
    pass


def normalize_walkability_score(score: float) -> float:
    """
    Normalize a walkability score to a 0-100 scale.

    Args:
        score: The raw walkability score

    Returns:
        Normalized score between 0 and 100

    Raises:
        WalkabilityScoreError: If score is negative
    """
    if score < 0:
        raise WalkabilityScoreError("Walkability score cannot be negative")

    # Clamp to 100 max
    return min(score, 100.0)


def categorize_walkability(score: float) -> str:
    """
    Categorize a walkability score into descriptive categories.

    Args:
        score: Walkability score (0-100)

    Returns:
        Category string: 'Car-Dependent', 'Somewhat Walkable',
        'Very Walkable', or 'Walker's Paradise'

    Raises:
        WalkabilityScoreError: If score is out of valid range
    """
    if score < 0 or score > 100:
        raise WalkabilityScoreError(f"Score {score} must be between 0 and 100")

    if score < 25:
        return "Car-Dependent"
    elif score < 50:
        return "Somewhat Walkable"
    elif score < 75:
        return "Very Walkable"
    else:
        return "Walker's Paradise"


def calculate_walkability_premium(
    walkable_price: float,
    non_walkable_price: float
) -> float:
    """
    Calculate the price premium percentage for walkable areas.

    Args:
        walkable_price: Average price in walkable area
        non_walkable_price: Average price in non-walkable area

    Returns:
        Premium as a percentage (e.g., 15.5 for 15.5% premium)

    Raises:
        WalkabilityScoreError: If prices are invalid
    """
    if walkable_price < 0 or non_walkable_price < 0:
        raise WalkabilityScoreError("Prices cannot be negative")

    if non_walkable_price == 0:
        raise WalkabilityScoreError("Non-walkable price cannot be zero")

    premium = ((walkable_price - non_walkable_price) / non_walkable_price) * 100
    return round(premium, 2)


def aggregate_neighborhood_scores(
    scores: List[float],
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Aggregate multiple walkability scores for a neighborhood.

    Args:
        scores: List of walkability scores
        weights: Optional weights for each score (must sum to 1.0)

    Returns:
        Dictionary with 'mean', 'median', 'min', 'max', and 'weighted_mean' (if weights provided)

    Raises:
        WalkabilityScoreError: If inputs are invalid
    """
    if not scores:
        raise WalkabilityScoreError("Scores list cannot be empty")

    if any(s < 0 or s > 100 for s in scores):
        raise WalkabilityScoreError("All scores must be between 0 and 100")

    result = {
        'mean': round(mean(scores), 2),
        'median': round(median(scores), 2),
        'min': min(scores),
        'max': max(scores)
    }

    if weights is not None:
        if len(weights) != len(scores):
            raise WalkabilityScoreError("Weights and scores must have same length")

        if not (0.99 <= sum(weights) <= 1.01):  # Allow small floating point errors
            raise WalkabilityScoreError("Weights must sum to 1.0")

        if any(w < 0 for w in weights):
            raise WalkabilityScoreError("Weights cannot be negative")

        weighted_mean = sum(s * w for s, w in zip(scores, weights))
        result['weighted_mean'] = round(weighted_mean, 2)

    return result


def calculate_price_per_walkability_point(
    price: float,
    walkability_score: float
) -> float:
    """
    Calculate the price per walkability point metric.

    Args:
        price: Housing price or rent
        walkability_score: Walkability score (0-100)

    Returns:
        Price per walkability point

    Raises:
        WalkabilityScoreError: If inputs are invalid
    """
    if price < 0:
        raise WalkabilityScoreError("Price cannot be negative")

    if walkability_score <= 0 or walkability_score > 100:
        raise WalkabilityScoreError("Walkability score must be between 0 and 100")

    return round(price / walkability_score, 2)


def find_best_value_neighborhoods(
    neighborhoods: List[Dict[str, float]],
    top_n: int = 5
) -> List[Dict[str, float]]:
    """
    Find neighborhoods with the best walkability-to-price ratio.

    Args:
        neighborhoods: List of dicts with 'name', 'price', and 'walkability_score'
        top_n: Number of top neighborhoods to return

    Returns:
        Sorted list of neighborhoods with best value (highest walkability, lowest price)

    Raises:
        WalkabilityScoreError: If inputs are invalid
    """
    if not neighborhoods:
        raise WalkabilityScoreError("Neighborhoods list cannot be empty")

    if top_n < 1:
        raise WalkabilityScoreError("top_n must be at least 1")

    # Validate all neighborhoods have required fields
    for hood in neighborhoods:
        if not all(key in hood for key in ['name', 'price', 'walkability_score']):
            raise WalkabilityScoreError(
                "Each neighborhood must have 'name', 'price', and 'walkability_score'"
            )

        if hood['price'] <= 0:
            raise WalkabilityScoreError(f"Price for {hood['name']} must be positive")

        if not (0 < hood['walkability_score'] <= 100):
            raise WalkabilityScoreError(
                f"Walkability score for {hood['name']} must be between 0 and 100"
            )

    # Calculate value score (higher walkability per dollar is better)
    scored_neighborhoods = []
    for hood in neighborhoods:
        value_score = hood['walkability_score'] / hood['price'] * 1000  # Scale up for readability
        scored_hood = hood.copy()
        scored_hood['value_score'] = round(value_score, 4)
        scored_neighborhoods.append(scored_hood)

    # Sort by value score descending
    sorted_neighborhoods = sorted(
        scored_neighborhoods,
        key=lambda x: x['value_score'],
        reverse=True
    )

    return sorted_neighborhoods[:top_n]
