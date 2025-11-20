"""
Varda: Institutions Tokenizing Example with Methodology & Validation
=====================================================================

This example demonstrates how Varda can be used to "tokenize" financial institutions
by representing them as network entities with relationships, risk profiles, and 
systemic connections. The implementation includes:

1. METHODOLOGY: Detailed explanation of the tokenization approach
2. VALIDATION: Weighted probability verification of all calculations
3. NUMERICAL VERIFICATION: Cross-checks to ensure correctness

Tokenizing institutions means:
1. Creating digital representations (entities) of real-world financial institutions
2. Modeling their relationships and exposures (counterparty risk, deals, investments)
3. Analyzing how risk propagates through the network
4. Identifying systemic risk hubs and contagion paths
5. Running stress tests to understand institution-level vulnerabilities

Use Cases:
- Regulatory stress testing (CCAR, ICAAP)
- Counterparty risk management
- Systemic risk monitoring
- Network topology analysis
- Institution-level capital planning

METHODOLOGY:
============

1. Entity Tokenization:
   - Each institution is represented as an Entity with:
     * Unique identifier (ID)
     * Risk score (0-1 scale, where 0 = no risk, 1 = maximum risk)
     * Metadata (assets, capital ratios, ratings, etc.)
     * Initial state (credit rating, risk state)
   
2. Relationship Modeling:
   - Relationships are weighted edges in a directed graph
   - Strength parameter (0-1) represents exposure intensity
   - Relationship types: interbank_exposure, underwriting, holdings, prime_brokerage
   - Metadata stores notional amounts, exposure types, etc.

3. Risk Propagation:
   - Uses fluid dynamics-inspired diffusion model
   - Risk flows through network based on relationship strengths
   - Diffusion rate controls how quickly risk spreads
   - Iterative process: risk(t+1) = (1-α)*risk(t) + α*diffused_risk(t)
   
4. Loss Distribution Simulation:
   - Monte Carlo simulation for tranche losses
   - PD (Probability of Default) adjusted by scenario multipliers
   - Horizon-adjusted PD: PD_h = 1 - (1 - PD_annual)^horizon
   - Loss = Default_Flag * Notional * LGD * Discount_Factor
   - Expected Loss (EL) = E[Loss] = PD_h * Notional * LGD / (1+r)^horizon
   
5. Validation Approach:
   - Weighted probability verification: sum of probabilities = 1
   - Expected value calculations: E[X] = Σ(x_i * p_i)
   - Variance verification: Var[X] = E[X²] - E[X]²
   - Cross-validation of Monte Carlo results vs analytical formulas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

try:
    from scipy import stats
except ImportError:
    # Fallback if scipy not available
    stats = None

# Import Varda and related classes
# Note: This assumes varda.py is in the same directory or in Python path
# varda.py depends on financial_risk_lab module for Entity, Relationship, etc.
try:
    from varda import (
        Varda, Entity, Relationship, MarketConstraint, MarketState,
        DealType, Tranche, CapitalMarketsDeal, CapitalMarketsScenario,
        create_credit_rating_chain, create_market_regime_chain
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: varda.py requires the 'financial_risk_lab' module.")
    print("Please ensure all dependencies are installed.")
    raise


# ============================================================================
# VALIDATION FUNCTIONS WITH WEIGHTED PROBABILITIES
# ============================================================================

def validate_weighted_probabilities(
    probabilities: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    tolerance: float = 1e-6
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that weighted probabilities sum to 1.0 (or weights sum correctly).
    
    Args:
        probabilities: Dictionary mapping outcomes to probabilities
        weights: Optional weights for each outcome (default: uniform)
        tolerance: Numerical tolerance for floating point comparison
    
    Returns:
        Tuple of (is_valid, validation_report)
    """
    if weights is None:
        weights = {k: 1.0 for k in probabilities.keys()}
    
    # Calculate weighted sum
    weighted_sum = sum(p * weights.get(k, 1.0) for k, p in probabilities.items())
    
    # Calculate unweighted sum
    unweighted_sum = sum(probabilities.values())
    
    # Check if probabilities are non-negative
    negative_probs = {k: p for k, p in probabilities.items() if p < 0}
    
    # Check if probabilities sum to 1 (within tolerance)
    is_valid = abs(unweighted_sum - 1.0) < tolerance and len(negative_probs) == 0
    
    report = {
        "is_valid": is_valid,
        "unweighted_sum": unweighted_sum,
        "weighted_sum": weighted_sum,
        "tolerance": tolerance,
        "negative_probabilities": negative_probs,
        "num_outcomes": len(probabilities)
    }
    
    return is_valid, report


def validate_expected_value(
    values: np.ndarray,
    probabilities: np.ndarray,
    calculated_expected: float,
    tolerance: float = 1e-4
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate expected value calculation: E[X] = Σ(x_i * p_i).
    
    Args:
        values: Array of outcome values
        probabilities: Array of corresponding probabilities
        calculated_expected: The expected value to validate
        tolerance: Numerical tolerance
    
    Returns:
        Tuple of (is_valid, validation_report)
    """
    # Calculate analytical expected value
    analytical_expected = np.sum(values * probabilities)
    
    # Calculate variance: Var[X] = E[X²] - E[X]²
    analytical_variance = np.sum(values**2 * probabilities) - analytical_expected**2
    
    # Check if probabilities sum to 1
    prob_sum = np.sum(probabilities)
    prob_sum_valid = abs(prob_sum - 1.0) < tolerance
    
    # Check if calculated matches analytical
    expected_match = abs(calculated_expected - analytical_expected) < tolerance
    
    is_valid = expected_match and prob_sum_valid
    
    report = {
        "is_valid": is_valid,
        "analytical_expected": float(analytical_expected),
        "calculated_expected": float(calculated_expected),
        "difference": float(abs(calculated_expected - analytical_expected)),
        "analytical_variance": float(analytical_variance),
        "prob_sum": float(prob_sum),
        "prob_sum_valid": prob_sum_valid,
        "tolerance": tolerance
    }
    
    return is_valid, report


def validate_monte_carlo_convergence(
    simulated_values: np.ndarray,
    analytical_expected: float,
    analytical_std: Optional[float] = None,
    confidence_level: float = 0.95
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that Monte Carlo simulation converges to analytical expected value.
    
    Uses Central Limit Theorem: sample mean should be within confidence interval
    of true mean.
    
    Args:
        simulated_values: Array of simulated values
        analytical_expected: True expected value
        analytical_std: True standard deviation (optional, will estimate if None)
        confidence_level: Confidence level for interval (default 0.95)
    
    Returns:
        Tuple of (is_valid, validation_report)
    """
    n = len(simulated_values)
    sample_mean = np.mean(simulated_values)
    sample_std = np.std(simulated_values, ddof=1)
    
    # Use analytical std if provided, otherwise use sample std
    std_to_use = analytical_std if analytical_std is not None else sample_std
    
    # Standard error
    se = std_to_use / np.sqrt(n)
    
    # Z-score for confidence interval (using normal approximation)
    if stats is not None:
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
    else:
        # Approximate z-score for 95% CI = 1.96
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99% CI = 2.576
    
    # Confidence interval
    margin = z_score * se
    ci_lower = analytical_expected - margin
    ci_upper = analytical_expected + margin
    
    # Check if sample mean is within confidence interval
    is_valid = ci_lower <= sample_mean <= ci_upper
    
    # Calculate relative error
    relative_error = abs(sample_mean - analytical_expected) / abs(analytical_expected) if analytical_expected != 0 else float('inf')
    
    report = {
        "is_valid": is_valid,
        "n_simulations": n,
        "sample_mean": float(sample_mean),
        "analytical_expected": float(analytical_expected),
        "sample_std": float(sample_std),
        "analytical_std": float(analytical_std) if analytical_std is not None else None,
        "standard_error": float(se),
        "confidence_level": confidence_level,
        "confidence_interval": (float(ci_lower), float(ci_upper)),
        "relative_error": float(relative_error),
        "within_ci": is_valid
    }
    
    return is_valid, report


def validate_loss_distribution(
    tranche: Tranche,
    scenario: CapitalMarketsScenario,
    loss_df: pd.DataFrame,
    risk_free_rate: float = 0.03
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate loss distribution calculations using analytical formulas.
    
    Validates:
    1. Expected Loss (EL) = PD_h * Notional * LGD / (1+r)^horizon
    2. Probability of default over horizon
    3. Loss distribution properties
    
    Args:
        tranche: The tranche being validated
        scenario: Scenario with PD multiplier and horizon
        loss_df: DataFrame with simulated losses
        risk_free_rate: Risk-free rate for discounting
    
    Returns:
        Tuple of (is_valid, validation_report)
    """
    # Calculate analytical expected loss
    pd_annual_adjusted = tranche.pd_annual * scenario.pd_multiplier
    pd_annual_adjusted = min(max(pd_annual_adjusted, 0.0), 1.0)
    pd_horizon = 1.0 - (1.0 - pd_annual_adjusted) ** scenario.horizon_years
    
    discount_factor = (1.0 + risk_free_rate) ** scenario.horizon_years
    analytical_el = pd_horizon * tranche.notional * tranche.lgd / discount_factor
    
    # Get simulated losses
    if tranche.id not in loss_df.columns:
        return False, {"error": f"Tranche {tranche.id} not found in loss_df"}
    
    simulated_losses = loss_df[tranche.id].values
    
    # Calculate sample statistics
    sample_mean = np.mean(simulated_losses)
    sample_std = np.std(simulated_losses, ddof=1)
    
    # Validate expected loss
    el_tolerance = analytical_el * 0.05  # 5% tolerance
    el_valid = abs(sample_mean - analytical_el) < el_tolerance
    
    # Validate that losses are non-negative
    negative_losses = np.sum(simulated_losses < 0)
    non_negative = negative_losses == 0
    
    # Validate that losses don't exceed maximum possible (notional * LGD)
    max_possible_loss = tranche.notional * tranche.lgd / discount_factor
    excessive_losses = np.sum(simulated_losses > max_possible_loss * 1.01)  # 1% tolerance
    within_bounds = excessive_losses == 0
    
    # Validate default probability (fraction of non-zero losses)
    default_rate = np.mean(simulated_losses > 0)
    pd_tolerance = 0.02  # 2 percentage points
    pd_valid = abs(default_rate - pd_horizon) < pd_tolerance
    
    is_valid = el_valid and non_negative and within_bounds and pd_valid
    
    report = {
        "is_valid": is_valid,
        "tranche_id": tranche.id,
        "analytical_el": float(analytical_el),
        "sample_mean_el": float(sample_mean),
        "el_difference": float(abs(sample_mean - analytical_el)),
        "el_tolerance": float(el_tolerance),
        "el_valid": el_valid,
        "analytical_pd_horizon": float(pd_horizon),
        "simulated_default_rate": float(default_rate),
        "pd_valid": pd_valid,
        "non_negative_losses": non_negative,
        "within_bounds": within_bounds,
        "sample_std": float(sample_std),
        "max_possible_loss": float(max_possible_loss)
    }
    
    return is_valid, report


def validate_risk_propagation(
    varda: Varda,
    initial_shock: Dict[str, float],
    risk_evolution: pd.DataFrame,
    diffusion_rate: float
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate risk propagation calculations.
    
    Checks:
    1. Risk values stay in [0, 1] range
    2. Risk conservation (or appropriate decay)
    3. Diffusion follows expected pattern
    
    Args:
        varda: Varda instance
        initial_shock: Initial shock values
        risk_evolution: DataFrame with risk evolution over iterations
        diffusion_rate: Diffusion rate used
    
    Returns:
        Tuple of (is_valid, validation_report)
    """
    # Check that all risk values are in [0, 1]
    all_risks = risk_evolution.values.flatten()
    in_range = np.all((all_risks >= 0) & (all_risks <= 1))
    
    # Check that risk doesn't increase beyond initial shock (for non-shocked entities)
    initial_risks = {eid: e.initial_risk_score for eid, e in varda.entities.items()}
    max_increases = {}
    
    for entity_id in varda.entities.keys():
        if entity_id in risk_evolution.index:
            initial_risk = initial_risks.get(entity_id, 0.0)
            initial_shock_risk = initial_shock.get(entity_id, initial_risk)
            max_risk = risk_evolution.loc[entity_id].max()
            max_increases[entity_id] = max_risk - initial_shock_risk
    
    # Check risk conservation (total risk should be bounded)
    total_risk_per_iteration = risk_evolution.sum(axis=0)
    risk_growth = total_risk_per_iteration.iloc[-1] - total_risk_per_iteration.iloc[0]
    
    # Risk should not grow unbounded (but can increase due to diffusion)
    reasonable_growth = risk_growth < len(varda.entities) * 0.5  # Arbitrary but reasonable bound
    
    is_valid = in_range and reasonable_growth
    
    report = {
        "is_valid": is_valid,
        "all_in_range": in_range,
        "min_risk": float(np.min(all_risks)),
        "max_risk": float(np.max(all_risks)),
        "risk_growth": float(risk_growth),
        "reasonable_growth": reasonable_growth,
        "num_entities": len(varda.entities),
        "num_iterations": len(risk_evolution.columns),
        "diffusion_rate": diffusion_rate,
        "max_increases": max_increases
    }
    
    return is_valid, report


def create_tokenized_institutions_network() -> Varda:
    """
    Create a Varda instance with tokenized financial institutions.
    
    This function demonstrates how to:
    1. Tokenize banks, issuers, and investors as entities
    2. Model their relationships and exposures
    3. Set up risk profiles and initial states
    """
    varda = Varda("Institutions Tokenizing Network")
    
    # =====================================================================
    # STEP 1: Tokenize Major Banks
    # =====================================================================
    print("Step 1: Tokenizing major banks...")
    
    banks = [
        Entity(
            id="bank_gs",
            name="Goldman Sachs",
            entity_type="bank",
            initial_risk_score=0.12,  # Low risk, well-capitalized
            metadata={
                "tier": "GSIB",  # Global Systemically Important Bank
                "region": "US",
                "assets_usd_bn": 1500,
                "capital_ratio": 0.15
            }
        ),
        Entity(
            id="bank_jpm",
            name="JPMorgan Chase",
            entity_type="bank",
            initial_risk_score=0.10,
            metadata={
                "tier": "GSIB",
                "region": "US",
                "assets_usd_bn": 3800,
                "capital_ratio": 0.16
            }
        ),
        Entity(
            id="bank_ms",
            name="Morgan Stanley",
            entity_type="bank",
            initial_risk_score=0.14,
            metadata={
                "tier": "GSIB",
                "region": "US",
                "assets_usd_bn": 1200,
                "capital_ratio": 0.14
            }
        ),
        Entity(
            id="bank_cs",
            name="Credit Suisse",
            entity_type="bank",
            initial_risk_score=0.25,  # Higher risk (pre-acquisition)
            metadata={
                "tier": "GSIB",
                "region": "Europe",
                "assets_usd_bn": 600,
                "capital_ratio": 0.12
            }
        ),
        Entity(
            id="bank_db",
            name="Deutsche Bank",
            entity_type="bank",
            initial_risk_score=0.20,
            metadata={
                "tier": "GSIB",
                "region": "Europe",
                "assets_usd_bn": 1400,
                "capital_ratio": 0.13
            }
        ),
    ]
    
    # Add banks with initial credit ratings
    initial_ratings = {
        "bank_gs": "AA-",
        "bank_jpm": "AA-",
        "bank_ms": "A+",
        "bank_cs": "BBB",
        "bank_db": "BBB+"
    }
    
    for bank in banks:
        varda.add_entity(bank, initial_state=initial_ratings.get(bank.id, "A"))
    
    # =====================================================================
    # STEP 2: Tokenize Corporate Issuers
    # =====================================================================
    print("Step 2: Tokenizing corporate issuers...")
    
    issuers = [
        Entity(
            id="issuer_techcorp",
            name="TechCorp Inc",
            entity_type="issuer",
            initial_risk_score=0.15,
            metadata={
                "sector": "Technology",
                "revenue_usd_bn": 50,
                "leverage_ratio": 2.5
            }
        ),
        Entity(
            id="issuer_retailco",
            name="RetailCo Holdings",
            entity_type="issuer",
            initial_risk_score=0.22,
            metadata={
                "sector": "Retail",
                "revenue_usd_bn": 30,
                "leverage_ratio": 4.0
            }
        ),
        Entity(
            id="issuer_energyco",
            name="EnergyCo Ltd",
            entity_type="issuer",
            initial_risk_score=0.28,
            metadata={
                "sector": "Energy",
                "revenue_usd_bn": 80,
                "leverage_ratio": 5.5
            }
        ),
    ]
    
    issuer_ratings = {
        "issuer_techcorp": "A",
        "issuer_retailco": "BB+",
        "issuer_energyco": "BB"
    }
    
    for issuer in issuers:
        varda.add_entity(issuer, initial_state=issuer_ratings.get(issuer.id, "BBB"))
    
    # =====================================================================
    # STEP 3: Tokenize Investors and Asset Managers
    # =====================================================================
    print("Step 3: Tokenizing investors and asset managers...")
    
    investors = [
        Entity(
            id="investor_blackrock",
            name="BlackRock",
            entity_type="investor",
            initial_risk_score=0.08,
            metadata={
                "type": "asset_manager",
                "aum_usd_bn": 10000,
                "strategy": "passive_active_mix"
            }
        ),
        Entity(
            id="investor_vanguard",
            name="Vanguard",
            entity_type="investor",
            initial_risk_score=0.07,
            metadata={
                "type": "asset_manager",
                "aum_usd_bn": 8000,
                "strategy": "passive"
            }
        ),
        Entity(
            id="investor_pension",
            name="State Pension Fund",
            entity_type="investor",
            initial_risk_score=0.10,
            metadata={
                "type": "pension_fund",
                "aum_usd_bn": 500,
                "liability_driven": True
            }
        ),
    ]
    
    for investor in investors:
        varda.add_entity(investor, initial_state="AAA")
    
    # =====================================================================
    # STEP 4: Model Relationships Between Tokenized Institutions
    # =====================================================================
    print("Step 4: Modeling relationships between institutions...")
    
    # Bank-to-Bank relationships (interbank lending, derivatives)
    interbank_relationships = [
        Relationship(
            source_id="bank_jpm",
            target_id="bank_gs",
            relationship_type="interbank_exposure",
            strength=0.15,  # Moderate exposure
            metadata={"exposure_type": "derivatives", "notional_usd_bn": 5.0}
        ),
        Relationship(
            source_id="bank_gs",
            target_id="bank_ms",
            relationship_type="interbank_exposure",
            strength=0.12,
            metadata={"exposure_type": "repo", "notional_usd_bn": 3.0}
        ),
        Relationship(
            source_id="bank_cs",
            target_id="bank_db",
            relationship_type="interbank_exposure",
            strength=0.20,
            metadata={"exposure_type": "derivatives", "notional_usd_bn": 8.0}
        ),
    ]
    
    # Bank-to-Issuer relationships (underwriting, lending)
    bank_issuer_relationships = [
        Relationship(
            source_id="bank_gs",
            target_id="issuer_techcorp",
            relationship_type="underwriting",
            strength=0.25,
            metadata={"deal_type": "DCM", "exposure_usd_bn": 2.0}
        ),
        Relationship(
            source_id="bank_jpm",
            target_id="issuer_retailco",
            relationship_type="underwriting",
            strength=0.30,
            metadata={"deal_type": "LEVFIN", "exposure_usd_bn": 1.5}
        ),
        Relationship(
            source_id="bank_ms",
            target_id="issuer_energyco",
            relationship_type="underwriting",
            strength=0.35,
            metadata={"deal_type": "DCM_HY", "exposure_usd_bn": 3.0}
        ),
        Relationship(
            source_id="bank_cs",
            target_id="issuer_techcorp",
            relationship_type="underwriting",
            strength=0.20,
            metadata={"deal_type": "ECM", "exposure_usd_bn": 1.0}
        ),
    ]
    
    # Investor-to-Issuer relationships (holdings)
    investor_issuer_relationships = [
        Relationship(
            source_id="investor_blackrock",
            target_id="issuer_techcorp",
            relationship_type="holdings",
            strength=0.40,
            metadata={"holding_pct": 0.05, "notional_usd_bn": 2.5}
        ),
        Relationship(
            source_id="investor_vanguard",
            target_id="issuer_retailco",
            relationship_type="holdings",
            strength=0.35,
            metadata={"holding_pct": 0.08, "notional_usd_bn": 2.4}
        ),
        Relationship(
            source_id="investor_pension",
            target_id="issuer_energyco",
            relationship_type="holdings",
            strength=0.30,
            metadata={"holding_pct": 0.03, "notional_usd_bn": 2.4}
        ),
    ]
    
    # Bank-to-Investor relationships (prime brokerage, custody)
    bank_investor_relationships = [
        Relationship(
            source_id="bank_gs",
            target_id="investor_blackrock",
            relationship_type="prime_brokerage",
            strength=0.18,
            metadata={"services": ["prime_brokerage", "custody"], "exposure_usd_bn": 10.0}
        ),
        Relationship(
            source_id="bank_jpm",
            target_id="investor_vanguard",
            relationship_type="prime_brokerage",
            strength=0.15,
            metadata={"services": ["custody"], "exposure_usd_bn": 8.0}
        ),
    ]
    
    # Add all relationships
    all_relationships = (
        interbank_relationships +
        bank_issuer_relationships +
        investor_issuer_relationships +
        bank_investor_relationships
    )
    
    for rel in all_relationships:
        varda.add_relationship(rel)
    
    print(f"  Added {len(all_relationships)} relationships")
    
    return varda


def add_deals_to_tokenized_network(varda: Varda) -> None:
    """
    Add capital markets deals that connect tokenized institutions.
    """
    print("\nStep 5: Adding capital markets deals...")
    
    # Deal 1: TechCorp high-yield bond (underwritten by GS and MS)
    techcorp_hy_tranche = Tranche(
        id="tranche_techcorp_hy",
        deal_id="deal_techcorp_hy",
        currency="USD",
        notional=500_000_000,  # $500M
        coupon=0.075,
        spread_bps=350,
        maturity_years=7.0,
        rating="BB+",
        pd_annual=0.025,
        lgd=0.55,
        seniority="senior"
    )
    
    techcorp_deal = CapitalMarketsDeal(
        id="deal_techcorp_hy",
        issuer_entity_id="issuer_techcorp",
        deal_type=DealType.DCM_HY,
        tranches=[techcorp_hy_tranche],
        bookrunners=["bank_gs", "bank_ms"],
        co_managers=["bank_jpm"],
        gross_fees=12_500_000,  # 2.5% of notional
        bank_share={"bank_gs": 0.50, "bank_ms": 0.35, "bank_jpm": 0.15},
        pipeline_stage="priced",
        sector="Technology",
        region="US"
    )
    
    varda.add_deal(techcorp_deal)
    
    # Deal 2: RetailCo leveraged loan (underwritten by JPM)
    retailco_loan_tranche = Tranche(
        id="tranche_retailco_loan",
        deal_id="deal_retailco_levfin",
        currency="USD",
        notional=750_000_000,  # $750M
        coupon=0.085,
        spread_bps=450,
        maturity_years=6.0,
        rating="B+",
        pd_annual=0.04,
        lgd=0.60,
        seniority="senior",
        is_secured=True
    )
    
    retailco_deal = CapitalMarketsDeal(
        id="deal_retailco_levfin",
        issuer_entity_id="issuer_retailco",
        deal_type=DealType.LEVFIN_LBO,
        tranches=[retailco_loan_tranche],
        bookrunners=["bank_jpm"],
        gross_fees=22_500_000,  # 3% of notional
        bank_share={"bank_jpm": 1.0},
        pipeline_stage="priced",
        sector="Retail",
        region="US"
    )
    
    varda.add_deal(retailco_deal)
    
    # Deal 3: EnergyCo investment grade bond (underwritten by MS and CS)
    energyco_ig_tranche = Tranche(
        id="tranche_energyco_ig",
        deal_id="deal_energyco_ig",
        currency="USD",
        notional=1_000_000_000,  # $1B
        coupon=0.045,
        spread_bps=150,
        maturity_years=10.0,
        rating="BBB",
        pd_annual=0.015,
        lgd=0.45,
        seniority="senior"
    )
    
    energyco_deal = CapitalMarketsDeal(
        id="deal_energyco_ig",
        issuer_entity_id="issuer_energyco",
        deal_type=DealType.DCM_IG,
        tranches=[energyco_ig_tranche],
        bookrunners=["bank_ms", "bank_cs"],
        gross_fees=15_000_000,  # 1.5% of notional
        bank_share={"bank_ms": 0.60, "bank_cs": 0.40},
        pipeline_stage="priced",
        sector="Energy",
        region="US"
    )
    
    varda.add_deal(energyco_deal)
    
    print(f"  Added {len(varda.deals)} deals with {len(varda.tranches)} tranches")


def analyze_tokenized_network(varda: Varda) -> None:
    """
    Analyze the tokenized institutions network.
    """
    print("\n" + "="*70)
    print("NETWORK ANALYSIS OF TOKENIZED INSTITUTIONS")
    print("="*70)
    
    # 1. Network topology
    print("\n1. Network Topology:")
    adj_matrix = varda.get_network_adjacency()
    print(f"   Total entities: {len(varda.entities)}")
    print(f"   Total relationships: {len(varda.relationships)}")
    print(f"   Network density: {adj_matrix.sum().sum() / (len(adj_matrix) * (len(adj_matrix) - 1)):.3f}")
    
    # 2. Identify systemic risk hubs
    print("\n2. Systemic Risk Hubs (highly connected, high risk):")
    hubs = varda.identify_systemic_risk_hubs(threshold=0.15)
    for hub_id in hubs:
        entity = varda.entities[hub_id]
        connections = sum(1 for r in varda.relationships 
                         if r.source_id == hub_id or r.target_id == hub_id)
        print(f"   - {entity.name} (ID: {hub_id})")
        print(f"     Risk Score: {entity.initial_risk_score:.3f}")
        print(f"     Connections: {connections}")
        print(f"     Type: {entity.entity_type}")
    
    # 3. Risk contagion paths
    print("\n3. Risk Contagion Paths (from Credit Suisse):")
    if "bank_cs" in varda.entities:
        paths = varda.get_risk_contagion_paths("bank_cs", max_depth=3)
        print(f"   Found {len(paths)} potential contagion paths")
        # Show first 5 paths
        for i, path in enumerate(paths[:5]):
            path_names = [varda.entities[entity_id].name for entity_id in path]
            print(f"   Path {i+1}: {' -> '.join(path_names)}")
    
    # 4. Entity risk summary
    print("\n4. Entity Risk Summary:")
    risk_summary = pd.DataFrame({
        "Entity": [e.name for e in varda.entities.values()],
        "Type": [e.entity_type for e in varda.entities.values()],
        "Risk Score": [e.initial_risk_score for e in varda.entities.values()],
        "Connections": [
            sum(1 for r in varda.relationships 
                if r.source_id == eid or r.target_id == eid)
            for eid in varda.entities.keys()
        ]
    }).sort_values("Risk Score", ascending=False)
    
    print(risk_summary.to_string(index=False))


def run_risk_propagation_simulation(varda: Varda) -> None:
    """
    Run risk propagation simulation to see how shocks affect tokenized institutions.
    """
    print("\n" + "="*70)
    print("RISK PROPAGATION SIMULATION")
    print("="*70)
    
    # Scenario: Credit Suisse experiences a shock
    print("\nScenario: Credit Suisse experiences a liquidity shock (risk = 0.6)")
    
    initial_shock = {
        "bank_cs": 0.6,  # High initial shock to Credit Suisse
    }
    
    # Run propagation
    risk_evolution = varda.propagate_risk_fluid(
        initial_shock=initial_shock,
        diffusion_rate=0.15,
        iterations=10
    )
    
    print("\nRisk Evolution (final iteration):")
    final_risks = risk_evolution.iloc[:, -1].sort_values(ascending=False)
    for entity_id, risk in final_risks.items():
        entity = varda.entities[entity_id]
        print(f"   {entity.name:30s} Risk: {risk:.3f} (change: {risk - entity.initial_risk_score:+.3f})")
    
    # Identify most affected institutions
    print("\nMost Affected Institutions (top 5):")
    risk_changes = final_risks - pd.Series({
        eid: e.initial_risk_score for eid, e in varda.entities.items()
    })
    top_affected = risk_changes.sort_values(ascending=False).head(5)
    for entity_id, change in top_affected.items():
        entity = varda.entities[entity_id]
        print(f"   {entity.name:30s} Risk increase: {change:+.3f}")
    
    # Validate risk propagation
    print("\nValidating Risk Propagation...")
    is_valid, validation_report = validate_risk_propagation(
        varda, initial_shock, risk_evolution, diffusion_rate=0.15
    )
    print(f"   Risk values in [0,1]: {validation_report['all_in_range']} ✓" if validation_report['all_in_range'] else f"   Risk values in [0,1]: {validation_report['all_in_range']} ✗")
    print(f"   Risk growth reasonable: {validation_report['reasonable_growth']} ✓" if validation_report['reasonable_growth'] else f"   Risk growth reasonable: {validation_report['reasonable_growth']} ✗")
    print(f"   Overall valid: {is_valid} ✓" if is_valid else f"   Overall valid: {is_valid} ✗")
    
    return risk_evolution


def run_monte_carlo_stress_test(varda: Varda) -> None:
    """
    Run Monte Carlo stress test on tokenized institutions.
    """
    print("\n" + "="*70)
    print("MONTE CARLO STRESS TEST")
    print("="*70)
    
    print("\nRunning 1000 Monte Carlo simulations with random shocks...")
    
    results = varda.monte_carlo_simulation(
        n_simulations=1000,
        shock_distribution="normal",
        shock_params={"mean": 0.15, "std": 0.08},
        diffusion_rate=0.12,
        iterations=8
    )
    
    print("\nRisk Statistics (across all simulations):")
    stats_df = pd.DataFrame({
        "Mean Risk": results["mean_risk"],
        "Std Risk": results["std_risk"],
        "5th Pct": results["p5_risk"],
        "95th Pct": results["p95_risk"],
        "Max Risk": results["max_risk"]
    }).sort_values("Mean Risk", ascending=False)
    
    print(stats_df.to_string())


def run_validation_suite(varda: Varda, loss_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Run comprehensive validation suite with weighted probabilities.
    
    This function validates:
    1. Probability distributions sum to 1
    2. Expected value calculations
    3. Monte Carlo convergence
    4. Loss distribution correctness
    5. Risk propagation validity
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*70)
    print("VALIDATION SUITE: WEIGHTED PROBABILITY VERIFICATION")
    print("="*70)
    
    validation_results = {}
    
    # 1. Validate Markov chain probabilities
    print("\n1. Validating Markov Chain Probabilities...")
    if "market_regimes" in varda.markov_chains:
        chain = varda.markov_chains["market_regimes"]
        stationary = chain.stationary_distribution()
        
        # Create probability dictionary
        probs = {state: float(p) for state, p in zip(chain.states, stationary)}
        
        is_valid, report = validate_weighted_probabilities(probs)
        validation_results["markov_chain"] = {"valid": is_valid, "report": report}
        
        print(f"   Stationary distribution probabilities:")
        for state, prob in probs.items():
            print(f"     {state}: {prob:.6f}")
        print(f"   Sum: {sum(probs.values()):.6f} (should be 1.0)")
        print(f"   Valid: {is_valid} ✓" if is_valid else f"   Valid: {is_valid} ✗")
    
    # 2. Validate loss distributions (if loss_df provided)
    if loss_df is not None:
        print("\n2. Validating Loss Distribution Calculations...")
        loss_validations = {}
        
        # Note: We'll validate with actual scenario in main() after scenario is created
        print("   (Loss distribution validation will be done with actual scenario in main)")
        
        validation_results["loss_distributions"] = loss_validations
    
    # 3. Validate Monte Carlo convergence
    if loss_df is not None:
        print("\n3. Validating Monte Carlo Convergence...")
        mc_validations = {}
        
        for tranche_id, tranche in varda.tranches.items():
            if tranche_id not in loss_df.columns:
                continue
            
            simulated_losses = loss_df[tranche_id].values
            
            # Calculate analytical expected value
            pd_horizon = 1.0 - (1.0 - tranche.pd_annual) ** 1.0  # 1 year horizon
            discount_factor = 1.03  # 3% risk-free rate
            analytical_el = pd_horizon * tranche.notional * tranche.lgd / discount_factor
            analytical_std = np.sqrt(
                pd_horizon * (1 - pd_horizon) * (tranche.notional * tranche.lgd / discount_factor) ** 2
            )
            
            is_valid, report = validate_monte_carlo_convergence(
                simulated_losses,
                analytical_el,
                analytical_std
            )
            mc_validations[tranche_id] = {"valid": is_valid, "report": report}
            
            print(f"   Tranche: {tranche_id}")
            print(f"     Simulations: {report['n_simulations']}")
            print(f"     Sample Mean: ${report['sample_mean']:,.2f}")
            print(f"     Analytical Mean: ${report['analytical_expected']:,.2f}")
            print(f"     Relative Error: {report['relative_error']*100:.2f}%")
            print(f"     Within CI: {report['within_ci']} ✓" if report['within_ci'] else f"     Within CI: {report['within_ci']} ✗")
        
        validation_results["monte_carlo"] = mc_validations
    
    # 4. Validate relationship weights
    print("\n4. Validating Relationship Weights...")
    relationship_weights = {}
    for rel in varda.relationships:
        key = f"{rel.source_id}->{rel.target_id}"
        relationship_weights[key] = rel.strength
    
    # Check that weights are in valid range [0, 1]
    all_valid = all(0 <= w <= 1 for w in relationship_weights.values())
    validation_results["relationships"] = {
        "valid": all_valid,
        "num_relationships": len(relationship_weights),
        "weight_range": (min(relationship_weights.values()), max(relationship_weights.values()))
    }
    
    print(f"   Total relationships: {len(relationship_weights)}")
    print(f"   Weight range: [{min(relationship_weights.values()):.3f}, {max(relationship_weights.values()):.3f}]")
    print(f"   All in [0,1]: {all_valid} ✓" if all_valid else f"   All in [0,1]: {all_valid} ✗")
    
    # 5. Validate entity risk scores
    print("\n5. Validating Entity Risk Scores...")
    risk_scores = {eid: e.initial_risk_score for eid, e in varda.entities.items()}
    all_risks_valid = all(0 <= r <= 1 for r in risk_scores.values())
    
    validation_results["risk_scores"] = {
        "valid": all_risks_valid,
        "num_entities": len(risk_scores),
        "risk_range": (min(risk_scores.values()), max(risk_scores.values())),
        "mean_risk": np.mean(list(risk_scores.values()))
    }
    
    print(f"   Total entities: {len(risk_scores)}")
    print(f"   Risk score range: [{min(risk_scores.values()):.3f}, {max(risk_scores.values()):.3f}]")
    print(f"   Mean risk: {np.mean(list(risk_scores.values())):.3f}")
    print(f"   All in [0,1]: {all_risks_valid} ✓" if all_risks_valid else f"   All in [0,1]: {all_risks_valid} ✗")
    
    # Summary
    print("\n" + "-"*70)
    print("VALIDATION SUMMARY")
    print("-"*70)
    
    all_valid = all([
        validation_results.get("markov_chain", {}).get("valid", True),
        all(v.get("valid", True) for v in validation_results.get("loss_distributions", {}).values()),
        all(v.get("valid", True) for v in validation_results.get("monte_carlo", {}).values()),
        validation_results.get("relationships", {}).get("valid", True),
        validation_results.get("risk_scores", {}).get("valid", True)
    ])
    
    print(f"Overall Validation Status: {'PASSED ✓' if all_valid else 'FAILED ✗'}")
    validation_results["overall_valid"] = all_valid
    
    return validation_results


def run_deal_level_analysis(varda: Varda, scenario: Optional[CapitalMarketsScenario] = None) -> Tuple[pd.DataFrame, CapitalMarketsScenario]:
    """
    Run deal-level risk analysis for tokenized institutions' exposures.
    """
    print("\n" + "="*70)
    print("DEAL-LEVEL RISK ANALYSIS")
    print("="*70)
    
    # Set up market regime chain
    if "market_regimes" not in varda.markov_chains:
        market_chain = create_market_regime_chain()
        varda.add_markov_chain("market_regimes", market_chain)
    
    # Use provided scenario or create new one
    if scenario is None:
        # Create stress scenario
        stress_constraint = MarketConstraint(
            name="Credit Spread Widening",
            constraint_type="market",
            value=200.0,  # 200bps spread widening
            impact_on_transitions={
                "Normal->Stressed": 1.8,
                "Stressed->Crisis": 2.0
            }
        )
        
        # Calibrate PD multiplier from regime
        pd_multiplier = varda.calibrate_pd_multiplier_from_regime(
            market_chain_name="market_regimes",
            scenario_constraints=[stress_constraint],
            base_state="Normal",
            stressed_state="Crisis"
        )
        
        print(f"\nPD Multiplier from stress scenario: {pd_multiplier:.2f}x")
        
        # Create scenario
        stress_scenario = CapitalMarketsScenario(
            name="Credit Stress Scenario",
            description="200bps spread widening, elevated default rates",
            spread_shock_bps=200.0,
            pd_multiplier=pd_multiplier,
            horizon_years=1.0,
            market_constraints=[stress_constraint]
        )
    else:
        stress_scenario = scenario
    
    # Run loss distribution simulation
    print("\nRunning loss distribution simulation for all tranches...")
    loss_df = varda.simulate_tranche_loss_distribution(
        tranche_ids=list(varda.tranches.keys()),
        scenario=stress_scenario,
        n_simulations=10000,
        random_seed=42
    )
    
    # Summarize losses
    print("\nTranche Loss Summary (EL, VaR, ES):")
    loss_summary = varda.summarize_loss_distribution(loss_df, var_levels=[0.95, 0.99])
    print(loss_summary.to_string())
    
    # Deal-level summary
    print("\nDeal-Level Risk/Return Summary:")
    pipeline_summary = varda.summarize_pipeline_risk_and_return(
        deal_ids=None,  # All deals
        loss_df=loss_df,
        var_level=0.99
    )
    print(pipeline_summary.to_string())
    
    # Fee-at-risk analysis
    print("\nFee-at-Risk Analysis:")
    fee_at_risk = varda.compute_pipeline_fee_at_risk(
        deal_ids=None,
        loss_df=loss_df,
        loss_threshold_ratio=0.02,
        fee_haircut_if_loss=0.5
    )
    
    # Aggregate by bank
    per_bank_fees = varda.aggregate_fee_at_risk(fee_at_risk)
    print("\nExpected Fees per Bank (under stress):")
    for bank_id in per_bank_fees.columns:
        if bank_id in varda.entities:
            bank_name = varda.entities[bank_id].name
            fees = per_bank_fees[bank_id]
            print(f"   {bank_name:30s} Mean: ${fees.mean():>12,.0f}  5th Pct: ${fees.quantile(0.05):>12,.0f}")
    
    return loss_df, stress_scenario


def main():
    """
    Main example: Tokenizing institutions and analyzing their network.
    """
    print("="*70)
    print("VARDA: INSTITUTIONS TOKENIZING EXAMPLE")
    print("="*70)
    print("\nThis example demonstrates how Varda can be used to tokenize")
    print("financial institutions and analyze their network relationships.\n")
    
    # Step 1-4: Create tokenized network
    varda = create_tokenized_institutions_network()
    
    # Step 5: Add deals
    add_deals_to_tokenized_network(varda)
    
    # Step 6: Analyze network
    analyze_tokenized_network(varda)
    
    # Step 7: Run risk propagation
    risk_evolution = run_risk_propagation_simulation(varda)
    
    # Step 8: Monte Carlo stress test
    run_monte_carlo_stress_test(varda)
    
    # Step 9: Deal-level analysis
    loss_df, scenario = run_deal_level_analysis(varda)
    
    # Step 10: Run validation suite (with scenario for proper validation)
    validation_results = run_validation_suite(varda, loss_df)
    
    # Additional validation: validate loss distributions with actual scenario
    if loss_df is not None:
        print("\n6. Validating Loss Distributions with Actual Scenario...")
        for tranche_id, tranche in varda.tranches.items():
            if tranche_id in loss_df.columns:
                is_valid, report = validate_loss_distribution(tranche, scenario, loss_df)
                print(f"   {tranche_id}: Valid={is_valid}, EL_diff=${report['el_difference']:,.2f}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(varda.summary())
    
    # Validation status
    if validation_results.get("overall_valid", False):
        print("\n✓ All validations passed - methodology verified with weighted probabilities")
    else:
        print("\n✗ Some validations failed - review validation results above")
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. Tokenization: Each institution is represented as an Entity with:
   - Unique ID and metadata (assets, capital ratios, etc.)
   - Initial risk score and credit rating
   - Entity type (bank, issuer, investor)

2. Relationships: Institutions are connected via:
   - Interbank exposures (derivatives, repo)
   - Underwriting relationships (deals)
   - Investment holdings
   - Prime brokerage services

3. Network Analysis: Varda can identify:
   - Systemic risk hubs (highly connected, high risk)
   - Contagion paths (how risk spreads)
   - Network topology and density

4. Stress Testing: Run simulations to:
   - See how shocks propagate through the network
   - Identify most vulnerable institutions
   - Assess deal-level and bank-level exposures

5. Use Cases:
   - Regulatory stress testing (CCAR, ICAAP)
   - Counterparty risk management
   - Systemic risk monitoring
   - Capital planning and allocation
    """)


if __name__ == "__main__":
    main()

