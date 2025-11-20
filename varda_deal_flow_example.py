"""
Varda Deal Flow Example: Token Analysis Through Transaction Lifecycle
======================================================================

This example demonstrates how Varda analyzes tokenized institutions
at each stage of a capital markets transaction, with the key equations
used for analysis.

Run this after setting up your Varda instance with institutions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from varda_institutions_tokenizing_demo import (
    Varda, Entity, Relationship, DealType, Tranche, CapitalMarketsDeal,
    CapitalMarketsScenario, MarketConstraint, create_market_regime_chain
)


# ============================================================================
# KEY EQUATIONS FOR TOKEN ANALYSIS
# ============================================================================

def calculate_network_centrality(varda: Varda, entity_id: str) -> float:
    """
    Equation 1: Network Centrality Score
    
    centrality = (in_degree + out_degree) / max_possible_connections
    
    Identifies how central an institution is in the network.
    """
    adj_matrix = varda.get_network_adjacency()
    
    if entity_id not in adj_matrix.index:
        return 0.0
    
    in_degree = adj_matrix[entity_id].sum()  # Incoming connections
    out_degree = adj_matrix.loc[entity_id].sum()  # Outgoing connections
    total_degree = in_degree + out_degree
    
    max_possible = 2 * (len(adj_matrix) - 1)  # Max connections (bidirectional)
    
    centrality = total_degree / max_possible if max_possible > 0 else 0.0
    return float(centrality)


def calculate_systemic_risk_contribution(
    varda: Varda, 
    entity_id: str,
    exposure_size: float = 1.0
) -> float:
    """
    Equation 2: Systemic Risk Contribution
    
    systemic_risk = risk_score × connectivity × exposure_size
    
    Measures how much an institution contributes to systemic risk.
    """
    if entity_id not in varda.entities:
        return 0.0
    
    entity = varda.entities[entity_id]
    risk_score = entity.initial_risk_score
    
    # Connectivity (number of connections)
    connections = sum(1 for r in varda.relationships 
                     if r.source_id == entity_id or r.target_id == entity_id)
    max_connections = len(varda.entities) - 1
    connectivity = connections / max_connections if max_connections > 0 else 0.0
    
    systemic_risk = risk_score * connectivity * exposure_size
    return float(systemic_risk)


def calculate_expected_loss(
    pd_annual: float,
    notional: float,
    lgd: float,
    horizon_years: float = 1.0,
    risk_free_rate: float = 0.03
) -> float:
    """
    Equation 3: Expected Loss (EL)
    
    EL = PD_horizon × Notional × LGD / (1 + r)^horizon
    
    Where:
    PD_horizon = 1 - (1 - PD_annual)^horizon
    """
    pd_horizon = 1.0 - (1.0 - pd_annual) ** horizon_years
    discount_factor = (1.0 + risk_free_rate) ** horizon_years
    el = pd_horizon * notional * lgd / discount_factor
    return float(el)


def calculate_var_and_es(
    loss_distribution: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    Equation 4: Value-at-Risk (VaR) and Expected Shortfall (ES)
    
    VaR_α = quantile(loss_distribution, α)
    ES_α = E[Loss | Loss ≥ VaR_α]
    """
    results = {}
    
    for alpha in confidence_levels:
        var = float(np.quantile(loss_distribution, alpha))
        es = float(np.mean(loss_distribution[loss_distribution >= var]))
        
        results[f"VaR_{int(alpha*100)}"] = var
        results[f"ES_{int(alpha*100)}"] = es
    
    return results


def calculate_raroc(
    fee_income: float,
    expected_loss: float,
    economic_capital: float
) -> float:
    """
    Equation 5: Risk-Adjusted Return on Capital (RAROC)
    
    RAROC = (Fee_Income - Expected_Loss) / Economic_Capital
    """
    if economic_capital <= 0:
        return float('inf') if (fee_income - expected_loss) > 0 else 0.0
    
    raroc = (fee_income - expected_loss) / economic_capital
    return float(raroc)


def calculate_network_risk_multiplier(
    varda: Varda,
    entity_id: str,
    connected_entity_risks: Optional[Dict[str, float]] = None
) -> float:
    """
    Equation 6: Network Risk Multiplier
    
    network_risk_multiplier = f(connected_entity_risks, relationship_strengths)
    
    Adjusts standalone risk for network effects.
    """
    if entity_id not in varda.entities:
        return 1.0
    
    # Get relationships
    relationships = [r for r in varda.relationships 
                   if r.source_id == entity_id or r.target_id == entity_id]
    
    if not relationships:
        return 1.0
    
    # Calculate weighted average risk of connected entities
    if connected_entity_risks is None:
        connected_entity_risks = {}
        for rel in relationships:
            connected_id = rel.target_id if rel.source_id == entity_id else rel.source_id
            if connected_id in varda.entities:
                connected_entity_risks[connected_id] = varda.entities[connected_id].initial_risk_score
    
    total_weighted_risk = 0.0
    total_weight = 0.0
    
    for rel in relationships:
        connected_id = rel.target_id if rel.source_id == entity_id else rel.source_id
        if connected_id in connected_entity_risks:
            weight = rel.strength
            risk = connected_entity_risks[connected_id]
            total_weighted_risk += weight * risk
            total_weight += weight
    
    avg_connected_risk = total_weighted_risk / total_weight if total_weight > 0 else 0.0
    
    # Network multiplier: increases with connected entity risk
    # Base case: multiplier = 1.0
    # If connected entities have high risk, multiplier > 1.0
    multiplier = 1.0 + 0.5 * avg_connected_risk  # Simple linear model
    
    return float(multiplier)


def calculate_concentration_risk(
    exposures: Dict[str, float]
) -> float:
    """
    Equation 7: Concentration Risk (Herfindahl-Hirschman Index)
    
    HHI = Σ(Exposure_i / Total_Exposure)^2
    Concentration_Risk = HHI × Average_PD
    """
    if not exposures:
        return 0.0
    
    total_exposure = sum(exposures.values())
    if total_exposure <= 0:
        return 0.0
    
    hhi = sum((exp / total_exposure) ** 2 for exp in exposures.values())
    return float(hhi)


# ============================================================================
# DEAL FLOW STAGE ANALYSIS
# ============================================================================

def analyze_stage_idea(varda: Varda, issuer_id: str) -> Dict[str, any]:
    """
    STAGE 1: IDEA - Analyze issuer's network position before engagement.
    """
    print("\n" + "="*70)
    print("STAGE 1: IDEA / PIPELINE BUILDING")
    print("="*70)
    
    if issuer_id not in varda.entities:
        raise ValueError(f"Issuer {issuer_id} not found")
    
    issuer = varda.entities[issuer_id]
    
    # Equation 1: Network Centrality
    centrality = calculate_network_centrality(varda, issuer_id)
    print(f"\n1. Network Centrality Score: {centrality:.4f}")
    print(f"   Interpretation: {'High' if centrality > 0.3 else 'Medium' if centrality > 0.1 else 'Low'} centrality")
    
    # Equation 2: Systemic Risk Contribution
    systemic_risk = calculate_systemic_risk_contribution(varda, issuer_id)
    print(f"\n2. Systemic Risk Contribution: {systemic_risk:.4f}")
    
    # Check if issuer is a systemic hub
    hubs = varda.identify_systemic_risk_hubs(threshold=0.15)
    is_hub = issuer_id in hubs
    print(f"   Is Systemic Risk Hub: {is_hub}")
    
    # Count connections
    connections = sum(1 for r in varda.relationships 
                    if r.source_id == issuer_id or r.target_id == issuer_id)
    print(f"   Number of Network Connections: {connections}")
    
    # Analyze connected entities
    connected_entities = []
    for rel in varda.relationships:
        if rel.source_id == issuer_id:
            connected_entities.append((rel.target_id, rel.strength, "outgoing"))
        elif rel.target_id == issuer_id:
            connected_entities.append((rel.source_id, rel.strength, "incoming"))
    
    print(f"\n3. Connected Entities ({len(connected_entities)}):")
    for entity_id, strength, direction in connected_entities[:5]:  # Show top 5
        if entity_id in varda.entities:
            entity = varda.entities[entity_id]
            print(f"   - {entity.name} ({direction}, strength: {strength:.3f})")
    
    return {
        "centrality": centrality,
        "systemic_risk": systemic_risk,
        "is_hub": is_hub,
        "connections": connections,
        "connected_entities": len(connected_entities)
    }


def analyze_stage_mandated(
    varda: Varda,
    deal: CapitalMarketsDeal,
    tranche: Tranche
) -> Dict[str, any]:
    """
    STAGE 2: MANDATED - Preliminary risk assessment.
    """
    print("\n" + "="*70)
    print("STAGE 2: MANDATED / ENGAGEMENT")
    print("="*70)
    
    # Equation 3: Expected Loss
    el = calculate_expected_loss(
        pd_annual=tranche.pd_annual,
        notional=tranche.notional,
        lgd=tranche.lgd,
        horizon_years=1.0
    )
    print(f"\n1. Expected Loss (EL): ${el:,.0f}")
    print(f"   Formula: EL = PD_horizon × Notional × LGD / (1+r)^horizon")
    
    pd_horizon = 1.0 - (1.0 - tranche.pd_annual) ** 1.0
    print(f"   PD_horizon = {pd_horizon:.4f} (from PD_annual = {tranche.pd_annual:.4f})")
    
    # Fee analysis
    fee_bps = deal.gross_fees / tranche.notional * 10_000
    print(f"\n2. Fee Analysis:")
    print(f"   Gross Fees: ${deal.gross_fees:,.0f}")
    print(f"   Fee (bps): {fee_bps:.1f}")
    print(f"   EL / Fees: {el / deal.gross_fees:.2f}x")
    
    # Bank capacity check
    print(f"\n3. Bank Capacity Analysis:")
    for bank_id in deal.bookrunners:
        if bank_id in varda.entities:
            bank = varda.entities[bank_id]
            bank_fee = deal.gross_fees * deal.bank_share.get(bank_id, 0.0)
            print(f"   {bank.name}:")
            print(f"     Fee Share: ${bank_fee:,.0f}")
            print(f"     Bank Risk Score: {bank.initial_risk_score:.3f}")
    
    return {
        "expected_loss": el,
        "fee_bps": fee_bps,
        "el_over_fees": el / deal.gross_fees
    }


def analyze_stage_launched(
    varda: Varda,
    deal: CapitalMarketsDeal,
    tranche: Tranche
) -> Dict[str, any]:
    """
    STAGE 3: LAUNCHED - Market regime and pricing analysis.
    """
    print("\n" + "="*70)
    print("STAGE 3: LAUNCHED / MARKETING")
    print("="*70)
    
    # Market regime analysis
    if "market_regimes" not in varda.markov_chains:
        market_chain = create_market_regime_chain()
        varda.add_markov_chain("market_regimes", market_chain)
    
    # Get current market state
    steady_state = varda.analyze_market_steady_state("market_regimes")
    print(f"\n1. Market Regime Probabilities:")
    for state, prob in steady_state["steady_state"].items():
        print(f"   {state}: {prob:.4f}")
    
    # PD multiplier from regime
    stress_constraint = MarketConstraint(
        name="Current Market Conditions",
        constraint_type="market",
        value=0.0,  # Adjust based on actual conditions
        impact_on_transitions={"Normal->Stressed": 1.2, "Stressed->Crisis": 1.3}
    )
    
    pd_mult = varda.calibrate_pd_multiplier_from_regime(
        "market_regimes",
        scenario_constraints=[stress_constraint],
        base_state="Normal",
        stressed_state="Crisis"
    )
    print(f"\n2. PD Multiplier from Market Regime: {pd_mult:.2f}x")
    print(f"   Adjusted PD: {tranche.pd_annual * pd_mult:.4f}")
    
    # Network risk multiplier
    network_mult = calculate_network_risk_multiplier(varda, deal.issuer_entity_id)
    print(f"\n3. Network Risk Multiplier: {network_mult:.2f}x")
    print(f"   Accounts for contagion from connected entities")
    
    # Adjusted EL
    adjusted_el = calculate_expected_loss(
        pd_annual=tranche.pd_annual * pd_mult,
        notional=tranche.notional,
        lgd=tranche.lgd
    ) * network_mult
    
    print(f"\n4. Network-Adjusted EL: ${adjusted_el:,.0f}")
    
    return {
        "pd_multiplier": pd_mult,
        "network_multiplier": network_mult,
        "adjusted_el": adjusted_el
    }


def analyze_stage_priced(
    varda: Varda,
    deal: CapitalMarketsDeal,
    tranche: Tranche,
    scenario: CapitalMarketsScenario
) -> Dict[str, any]:
    """
    STAGE 4: PRICED - Full risk quantification with Monte Carlo.
    """
    print("\n" + "="*70)
    print("STAGE 4: PRICED / EXECUTION")
    print("="*70)
    
    # Full loss distribution
    print("\n1. Running Monte Carlo Loss Distribution...")
    loss_df = varda.simulate_tranche_loss_distribution(
        tranche_ids=[tranche.id],
        scenario=scenario,
        n_simulations=10000,
        random_seed=42
    )
    
    losses = loss_df[tranche.id].values
    
    # Equation 3: Expected Loss
    el = float(np.mean(losses))
    print(f"\n2. Expected Loss (EL): ${el:,.0f}")
    
    # Equation 4: VaR and ES
    var_es = calculate_var_and_es(losses, confidence_levels=[0.95, 0.99])
    print(f"\n3. Risk Metrics:")
    print(f"   VaR_95: ${var_es['VaR_95']:,.0f}")
    print(f"   ES_95: ${var_es['ES_95']:,.0f}")
    print(f"   VaR_99: ${var_es['VaR_99']:,.0f}")
    print(f"   ES_99: ${var_es['ES_99']:,.0f}")
    
    # Equation 5: RAROC
    economic_capital = var_es['ES_99'] - el  # ES - EL as proxy for EC
    raroc = calculate_raroc(deal.gross_fees, el, economic_capital)
    print(f"\n4. Risk-Adjusted Return on Capital (RAROC):")
    print(f"   Economic Capital: ${economic_capital:,.0f}")
    print(f"   RAROC: {raroc:.2%}")
    print(f"   Formula: RAROC = (Fee - EL) / EC")
    
    # Risk-return check
    var_over_fees = var_es['VaR_99'] / deal.gross_fees
    print(f"\n5. Risk-Return Analysis:")
    print(f"   VaR_99 / Fees: {var_over_fees:.2f}x")
    if var_over_fees > 2.0:
        print(f"   ⚠ WARNING: Tail risk exceeds 2x fees")
    else:
        print(f"   ✓ Risk-return profile acceptable")
    
    return {
        "el": el,
        "var_95": var_es['VaR_95'],
        "var_99": var_es['VaR_99'],
        "es_99": var_es['ES_99'],
        "economic_capital": economic_capital,
        "raroc": raroc
    }


def analyze_stage_closed(
    varda: Varda,
    deal: CapitalMarketsDeal,
    issuer_id: str
) -> Dict[str, any]:
    """
    STAGE 5: CLOSED - Ongoing monitoring and portfolio aggregation.
    """
    print("\n" + "="*70)
    print("STAGE 5: CLOSED / ON-BOOK")
    print("="*70)
    
    # Rating evolution simulation
    print("\n1. Simulating Rating Evolution...")
    # Note: Requires credit rating chain to be set up
    # This is a placeholder - in practice you'd have a rating chain
    
    # Concentration risk
    print("\n2. Portfolio Concentration Analysis:")
    bank_exposures = {}
    for bank_id in deal.bookrunners:
        if bank_id in varda.entities:
            # Sum all deals for this bank
            total_exposure = 0.0
            for d in varda.deals.values():
                if bank_id in d.bookrunners:
                    total_exposure += sum(t.notional for t in d.tranches) * d.bank_share.get(bank_id, 0.0)
            bank_exposures[bank_id] = total_exposure
    
    hhi = calculate_concentration_risk(bank_exposures)
    print(f"   Herfindahl-Hirschman Index (HHI): {hhi:.4f}")
    print(f"   Interpretation: {'High' if hhi > 0.25 else 'Medium' if hhi > 0.15 else 'Low'} concentration")
    
    # Network monitoring
    print("\n3. Network Position Monitoring:")
    centrality = calculate_network_centrality(varda, issuer_id)
    systemic_risk = calculate_systemic_risk_contribution(varda, issuer_id)
    print(f"   Current Centrality: {centrality:.4f}")
    print(f"   Current Systemic Risk: {systemic_risk:.4f}")
    
    # Check if issuer became a hub
    hubs = varda.identify_systemic_risk_hubs(threshold=0.15)
    if issuer_id in hubs:
        print(f"   ⚠ WARNING: Issuer has become a systemic risk hub")
    
    return {
        "concentration_hhi": hhi,
        "centrality": centrality,
        "systemic_risk": systemic_risk,
        "is_hub": issuer_id in hubs
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def demonstrate_deal_flow_analysis(varda: Varda):
    """
    Demonstrate token analysis through complete deal flow.
    """
    print("="*70)
    print("VARDA DEAL FLOW ANALYSIS: TOKEN ANALYSIS THROUGH TRANSACTION LIFECYCLE")
    print("="*70)
    
    # Assume we have a deal in the pipeline
    issuer_id = "issuer_techcorp"
    
    # STAGE 1: IDEA
    idea_analysis = analyze_stage_idea(varda, issuer_id)
    
    # Create a deal
    tranche = Tranche(
        id="tranche_demo",
        deal_id="deal_demo",
        currency="USD",
        notional=500_000_000,
        coupon=0.075,
        spread_bps=350,
        maturity_years=7.0,
        rating="BB+",
        pd_annual=0.025,
        lgd=0.55
    )
    
    deal = CapitalMarketsDeal(
        id="deal_demo",
        issuer_entity_id=issuer_id,
        deal_type=DealType.DCM_HY,
        tranches=[tranche],
        bookrunners=["bank_gs", "bank_jpm"],
        gross_fees=12_500_000,
        bank_share={"bank_gs": 0.60, "bank_jpm": 0.40},
        pipeline_stage="mandated"
    )
    
    # STAGE 2: MANDATED
    mandated_analysis = analyze_stage_mandated(varda, deal, tranche)
    
    # STAGE 3: LAUNCHED
    deal.pipeline_stage = "launched"
    launched_analysis = analyze_stage_launched(varda, deal, tranche)
    
    # STAGE 4: PRICED
    deal.pipeline_stage = "priced"
    varda.add_deal(deal)
    
    scenario = CapitalMarketsScenario(
        name="Pricing Validation",
        description="Final risk assessment",
        pd_multiplier=launched_analysis["pd_multiplier"],
        horizon_years=1.0
    )
    
    priced_analysis = analyze_stage_priced(varda, deal, tranche, scenario)
    
    # STAGE 5: CLOSED
    deal.pipeline_stage = "closed"
    closed_analysis = analyze_stage_closed(varda, deal, issuer_id)
    
    # Summary
    print("\n" + "="*70)
    print("DEAL FLOW SUMMARY")
    print("="*70)
    print(f"\nStage 1 (Idea): Centrality = {idea_analysis['centrality']:.4f}, Systemic Risk = {idea_analysis['systemic_risk']:.4f}")
    print(f"Stage 2 (Mandated): EL = ${mandated_analysis['expected_loss']:,.0f}, EL/Fees = {mandated_analysis['el_over_fees']:.2f}x")
    print(f"Stage 3 (Launched): PD Multiplier = {launched_analysis['pd_multiplier']:.2f}x, Network Multiplier = {launched_analysis['network_multiplier']:.2f}x")
    print(f"Stage 4 (Priced): VaR_99 = ${priced_analysis['var_99']:,.0f}, RAROC = {priced_analysis['raroc']:.2%}")
    print(f"Stage 5 (Closed): HHI = {closed_analysis['concentration_hhi']:.4f}, Is Hub = {closed_analysis['is_hub']}")


if __name__ == "__main__":
    # This would be run after creating your Varda instance with institutions
    # from varda_institutions_tokenizing_demo import create_tokenized_institutions_network
    # varda = create_tokenized_institutions_network()
    # demonstrate_deal_flow_analysis(varda)
    
    print("This module provides deal flow analysis functions.")
    print("Import and use with your Varda instance:")
    print("  from varda_deal_flow_example import demonstrate_deal_flow_analysis")
    print("  demonstrate_deal_flow_analysis(varda)")

