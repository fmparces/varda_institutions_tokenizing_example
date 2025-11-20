"""
================================================================================
DISCLAIMER: USE CASE DEMONSTRATION - NOT PRODUCTION READY
================================================================================

This file demonstrates the capabilities of the Varda Capital Markets Network 
Risk Lab for tokenizing financial institutions and analyzing systemic risk.

PURPOSE:
--------
This is a USE CASE DEMONSTRATION showing:
1. How to tokenize financial institutions as network entities
2. How to model relationships and exposures between institutions
3. How to run risk propagation simulations
4. How to validate calculations using weighted probabilities
5. How to perform Monte Carlo stress testing

AREAS FOR IMPROVEMENT (Future Work):
------------------------------------
1. **Data Quality & Calibration:**
   - Real-world institution data (actual balance sheets, exposures)
   - Calibrated PD/LGD models from historical defaults
   - Market-implied credit spreads and ratings
   - Dynamic relationship strengths based on market conditions

2. **Model Sophistication:**
   - More sophisticated correlation structures (copulas, factor models)
   - Time-varying risk propagation (not just static diffusion)
   - Liquidity risk modeling (funding liquidity, market liquidity)
   - Counterparty credit risk (CVA, DVA, FVA)

3. **Network Modeling:**
   - Multi-layer networks (credit, funding, derivatives)
   - Dynamic network topology (relationships change over time)
   - Feedback loops and non-linear effects
   - Network centrality measures beyond simple connectivity

4. **Validation & Backtesting:**
   - Historical backtesting against actual crisis events
   - Out-of-sample validation
   - Sensitivity analysis and stress testing framework
   - Model risk quantification

5. **Regulatory Compliance:**
   - CCAR/DFAST stress testing framework integration
   - ICAAP capital planning integration
   - IFRS 9 / CECL provisioning integration
   - Regulatory reporting formats

6. **Performance & Scalability:**
   - Optimization for large networks (1000+ entities)
   - Parallel/distributed Monte Carlo simulations
   - Real-time risk monitoring capabilities
   - API integration with trading systems

7. **User Interface:**
   - Interactive dashboards
   - Visualization of network topology and risk flows
   - Scenario builder GUI
   - Report generation

CURRENT LIMITATIONS:
--------------------
- Simplified risk propagation model (linear diffusion)
- Static network topology
- Basic correlation assumptions
- Limited validation against real-world data
- No real-time data integration
- Simplified loss distribution models

USE AT YOUR OWN RISK:
---------------------
This code is provided for demonstration and educational purposes only.
It should NOT be used for actual risk management, capital planning, or
regulatory reporting without significant enhancements and validation.

For production use, please:
1. Validate all models against historical data
2. Calibrate parameters using real-world data
3. Implement proper error handling and logging
4. Add comprehensive unit and integration tests
5. Conduct model risk assessment
6. Obtain appropriate regulatory approvals

================================================================================
VARDA: CAPITAL MARKETS NETWORK RISK LAB
Institutions Tokenizing Use Case with Methodology & Validation
================================================================================

This demonstration shows how Varda can be used to "tokenize" financial 
institutions by representing them as network entities with relationships, 
risk profiles, and systemic connections.

The implementation includes:
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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from scipy import stats
except ImportError:
    # Fallback if scipy not available
    stats = None

# Import base functionality from financial_risk_lab
# NOTE: This requires the financial_risk_lab module to be installed
try:
    from financial_risk_lab import (  # type: ignore
        Entity, Relationship, MarketConstraint, MarketState, MarkovChain,
        RiskType, create_credit_rating_chain, create_risk_state_chain,
        create_market_regime_chain
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This requires the 'financial_risk_lab' module.")
    print("Please ensure all dependencies are installed.")
    raise


# ============================================================================
# VARDA CORE CLASSES AND FUNCTIONALITY
# ============================================================================

class DealType(Enum):
    """Types of capital markets deals."""
    ECM_IPO = "ecm_ipo"
    ECM_FOLLOW_ON = "ecm_follow_on"
    DCM_IG = "dcm_investment_grade"
    DCM_HY = "dcm_high_yield"
    LEVFIN_LBO = "levfin_lbo"
    LEVFIN_REF = "levfin_refinancing"


@dataclass
class Tranche:
    """
    Represents a capital markets tranche (bond, loan, equity slice).

    For DCM/LevFin, PD/LGD/EAD are key for loss; for ECM, PD~0 but
    you care about price support / stabilization and overhang.
    """
    id: str
    deal_id: str
    currency: str
    notional: float       # EAD
    coupon: float         # as % (e.g., 0.05 = 5%)
    spread_bps: float     # vs benchmark
    maturity_years: float
    rating: str
    pd_annual: float      # annual default probability (approx)
    lgd: float            # loss given default (0-1)
    is_secured: bool = False
    seniority: str = "senior"  # senior, mezz, junior, equity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate tranche data."""
        if self.notional < 0:
            warnings.warn(f"Notional for {self.id} should be >= 0, got {self.notional}")
        if self.spread_bps < 0:
            warnings.warn(f"Spread for {self.id} should be >= 0 bps, got {self.spread_bps}")
        if not 0.0 <= self.pd_annual <= 1.0:
            warnings.warn(f"PD for {self.id} should be in [0, 1], got {self.pd_annual}")
        if not 0.0 <= self.lgd <= 1.0:
            warnings.warn(f"LGD for {self.id} should be in [0, 1], got {self.lgd}")


@dataclass
class CapitalMarketsDeal:
    """
    Represents an ECM/DCM/LevFin transaction.

    This ties your generic Entity/Relationship world to concrete
    capital-markets objects (underwriting, fees, tranches).
    """
    id: str
    issuer_entity_id: str
    deal_type: DealType
    tranches: List[Tranche]
    bookrunners: List[str]       # entity_ids of banks
    co_managers: List[str] = field(default_factory=list)
    issue_date: Optional[pd.Timestamp] = None
    tenor_years: Optional[float] = None
    gross_fees: float = 0.0      # total fee pool in currency
    bank_share: Dict[str, float] = field(default_factory=dict)  # bank_id -> fee share (0-1)
    pipeline_stage: str = "mandated"   # idea, mandated, launched, priced, closed
    sector: Optional[str] = None
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate deal data."""
        if self.bank_share:
            total_share = sum(self.bank_share.values())
            if abs(total_share - 1.0) > 0.01:  # Allow small rounding errors
                warnings.warn(f"Bank share sum should be ~1.0, got {total_share}")


@dataclass
class CapitalMarketsScenario:
    """
    Scenario for capital markets analysis.

    Combines macro/market constraints with spread/PD shocks and valuation inputs.

    Typical usage in a big-bank style workflow:
    - Derive pd_multiplier from a market regime Markov chain using
      Varda.calibrate_pd_multiplier_from_regime(...)
    - Attach MarketConstraint objects representing macro/market stress.
    """
    name: str
    description: str
    market_constraints: List[MarketConstraint] = field(default_factory=list)
    spread_shock_bps: float = 0.0          # parallel credit spread shock
    pd_multiplier: float = 1.0             # multiply PDs by this factor
    equity_vol_multiplier: float = 1.0     # for ECM price paths
    discount_rate_shift_bps: float = 0.0   # shift in discount rate
    horizon_years: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Varda:
    """
    Main Varda platform class for capital markets risk modeling and simulation.
    
    Varda models systemic risk, contagion, and credit scenarios for ECM, DCM, and LevFin using:
    - Network models to represent entity relationships (issuers, banks, investors)
    - Deal-aware modeling of capital markets transactions
    - Fluid dynamics metaphors for risk propagation
    - Markov chain models for state transitions (credit ratings, risk states)
    - Monte Carlo simulations for scenario analysis and loss distributions
    - Pipeline-level fee-at-risk and P&L analytics

    Outputs can be mapped to standard risk measures (EL, VaR, ES) and used to
    support internal stress tests and capital planning (e.g., CCAR / ICAAP / IFRS9 / CECL inputs).
    """
    
    def __init__(self, name: str = "Varda Capital Markets Risk Lab") -> None:
        """Initialize the Varda platform."""
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.simulation_history: List[Dict[str, Any]] = []
        self.markov_chains: Dict[str, MarkovChain] = {}
        self.entity_states: Dict[str, str] = {}
        self.market_states: Dict[str, MarketState] = {}
        self.market_constraints: List[MarketConstraint] = []
        self.deals: Dict[str, CapitalMarketsDeal] = {}
        self.tranches: Dict[str, Tranche] = {}
        
    def add_entity(self, entity: Entity, initial_state: Optional[str] = None) -> None:
        """Add an entity to the network."""
        self.entities[entity.id] = entity
        if initial_state is not None:
            self.entity_states[entity.id] = initial_state
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between entities."""
        if relationship.source_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_id} not found")
        if relationship.target_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_id} not found")
        self.relationships.append(relationship)
    
    def add_deal(self, deal: CapitalMarketsDeal) -> None:
        """Register a capital markets deal and its tranches."""
        if deal.issuer_entity_id not in self.entities:
            raise ValueError(f"Issuer entity {deal.issuer_entity_id} not found in Varda.entities")
        self.deals[deal.id] = deal
        for tranche in deal.tranches:
            if tranche.id in self.tranches:
                warnings.warn(f"Tranche {tranche.id} already exists. Overwriting.")
            self.tranches[tranche.id] = tranche
    
    def get_network_adjacency(self) -> pd.DataFrame:
        """Build adjacency matrix representing the entity network."""
        entity_ids = list(self.entities.keys())
        n = len(entity_ids)
        adj_matrix = np.zeros((n, n))
        id_to_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
        
        for rel in self.relationships:
            source_idx = id_to_idx[rel.source_id]
            target_idx = id_to_idx[rel.target_id]
            adj_matrix[source_idx, target_idx] = rel.strength
            
        return pd.DataFrame(adj_matrix, index=entity_ids, columns=entity_ids)
    
    def propagate_risk_fluid(
        self,
        initial_shock: Optional[Dict[str, float]] = None,
        diffusion_rate: float = 0.1,
        iterations: int = 10
    ) -> pd.DataFrame:
        """
        Simulate risk propagation using fluid dynamics-inspired diffusion model.
        Risk flows through the network like a fluid, with diffusion based on
        relationship strengths and connection topology.
        """
        entity_ids = list(self.entities.keys())
        n = len(entity_ids)
        risk_levels = np.zeros((iterations + 1, n))
        
        # Set initial conditions
        if initial_shock:
            for idx, entity_id in enumerate(entity_ids):
                risk_levels[0, idx] = initial_shock.get(entity_id, self.entities[entity_id].initial_risk_score)
        else:
            for idx, entity_id in enumerate(entity_ids):
                risk_levels[0, idx] = self.entities[entity_id].initial_risk_score
        
        # Get adjacency matrix and normalize
        adj_matrix = self.get_network_adjacency().values
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_adj = adj_matrix / row_sums
        
        # Propagate risk through iterations
        for t in range(iterations):
            diffused = normalized_adj @ risk_levels[t]
            risk_levels[t + 1] = (1 - diffusion_rate) * risk_levels[t] + diffusion_rate * diffused
            risk_levels[t + 1] = np.clip(risk_levels[t + 1], 0, 1)
        
        columns = [f"iteration_{i}" for i in range(iterations + 1)]
        return pd.DataFrame(risk_levels.T, index=entity_ids, columns=columns)
    
    def calibrate_pd_multiplier_from_regime(
        self,
        market_chain_name: str,
        scenario_constraints: List[MarketConstraint],
        base_state: str = "Normal",
        stressed_state: str = "Crisis"
    ) -> float:
        """Derive a PD multiplier from how much the steady-state probability of a
        stressed regime increases relative to the baseline."""
        if market_chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{market_chain_name}' not found")

        chain = self.markov_chains[market_chain_name]
        unconstrained = chain.stationary_distribution()
        constrained, _ = chain.constrained_stationary_distribution(
            scenario_constraints,
            state_names=chain.states
        )

        base_idx = chain.state_to_idx.get(base_state)
        stress_idx = chain.state_to_idx.get(stressed_state)
        if base_idx is None or stress_idx is None:
            raise ValueError(f"States '{base_state}' or '{stressed_state}' not found in market chain")

        base_crisis_prob = unconstrained[stress_idx]
        stressed_crisis_prob = constrained[stress_idx]
        ratio = (stressed_crisis_prob + 1e-6) / (base_crisis_prob + 1e-6)
        return float(max(1.0, ratio))
    
    def simulate_tranche_loss_distribution(
        self,
        tranche_ids: List[str],
        scenario: CapitalMarketsScenario,
        n_simulations: int = 10000,
        risk_free_rate: float = 0.03,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Monte Carlo loss distribution for a set of DCM/LevFin tranches."""
        if random_seed is not None:
            np.random.seed(random_seed)

        records: List[pd.Series] = []
        effective_rf = risk_free_rate + scenario.discount_rate_shift_bps / 10000.0
        horizon = scenario.horizon_years

        for tranche_id in tranche_ids:
            if tranche_id not in self.tranches:
                raise ValueError(f"Tranche {tranche_id} not found")
            tr = self.tranches[tranche_id]

            pd_annual = tr.pd_annual * scenario.pd_multiplier
            pd_annual = min(max(pd_annual, 0.0), 1.0)
            pd_horizon = 1.0 - (1.0 - pd_annual) ** horizon
            lgd = tr.lgd
            notional = tr.notional

            default_draws = np.random.rand(n_simulations)
            default_flags = (default_draws < pd_horizon).astype(float)

            cash_loss = default_flags * notional * lgd
            discounted_loss = cash_loss / ((1.0 + effective_rf) ** horizon)

            records.append(pd.Series(discounted_loss, name=tranche_id))

        loss_df = pd.concat(records, axis=1)
        loss_df.attrs["scenario"] = scenario.name
        loss_df.attrs["n_simulations"] = n_simulations
        self.simulation_history.append({
            "type": "tranche_loss",
            "scenario": scenario.name,
            "loss_df": loss_df
        })
        return loss_df
    
    def summarize_loss_distribution(
        self,
        loss_df: pd.DataFrame,
        var_levels: List[float] = [0.95, 0.99]
    ) -> pd.DataFrame:
        """Summarize loss distributions with EL, UL, VaR, and ES, per tranche."""
        summary_records: List[Dict[str, Any]] = []
        for col in loss_df.columns:
            losses = loss_df[col]
            el = float(losses.mean())
            ul = float(losses.std())
            rec: Dict[str, Any] = {"tranche_id": col, "EL": el, "UL": ul}
            for q in var_levels:
                var = float(losses.quantile(q))
                es = float(losses[losses >= var].mean())
                rec[f"VaR_{int(q * 100)}"] = var
                rec[f"ES_{int(q * 100)}"] = es
            summary_records.append(rec)
        return pd.DataFrame(summary_records).set_index("tranche_id")
    
    def summarize_deal_risk_and_return(
        self,
        deal_id: str,
        loss_df: pd.DataFrame,
        var_level: float = 0.99
    ) -> Dict[str, Any]:
        """Summarize a single deal's underwriting risk vs economics."""
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]
        deal_tranche_ids = [t.id for t in deal.tranches if t.id in loss_df.columns]
        if not deal_tranche_ids:
            raise ValueError(f"No tranche losses found in loss_df for deal {deal_id}")

        deal_losses = loss_df[deal_tranche_ids].sum(axis=1)
        el = float(deal_losses.mean())
        var = float(deal_losses.quantile(var_level))
        es = float(deal_losses[deal_losses >= var].mean())

        total_notional = float(sum(self.tranches[t].notional for t in deal_tranche_ids))
        total_notional = max(total_notional, 1e-6)
        gross_fees = float(deal.gross_fees)

        fee_bps = gross_fees / total_notional * 10_000.0 if total_notional > 0 else 0.0
        el_pct_notional = el / total_notional if total_notional > 0 else 0.0
        var_pct_notional = var / total_notional if total_notional > 0 else 0.0
        el_over_fees = el / gross_fees if gross_fees > 0 else np.nan
        var_over_fees = var / gross_fees if gross_fees > 0 else np.nan

        return {
            "deal_id": deal_id,
            "deal_type": deal.deal_type.value,
            "pipeline_stage": deal.pipeline_stage,
            "sector": deal.sector,
            "region": deal.region,
            "notional": total_notional,
            "gross_fees": gross_fees,
            "fee_bps": fee_bps,
            "EL": el,
            "VaR": var,
            "ES": es,
            "EL_pct_notional": el_pct_notional,
            "VaR_pct_notional": var_pct_notional,
            "EL_over_fees": el_over_fees,
            "VaR_over_fees": var_over_fees,
        }

    def summarize_pipeline_risk_and_return(
        self,
        deal_ids: Optional[List[str]],
        loss_df: pd.DataFrame,
        var_level: float = 0.99
    ) -> pd.DataFrame:
        """Summarize risk/return metrics across a set of deals."""
        if deal_ids is None:
            deal_ids = []
            for deal_id, deal in self.deals.items():
                if any(t.id in loss_df.columns for t in deal.tranches):
                    deal_ids.append(deal_id)

        records: List[Dict[str, Any]] = []
        for d_id in deal_ids:
            try:
                rec = self.summarize_deal_risk_and_return(d_id, loss_df, var_level)
                records.append(rec)
            except ValueError:
                continue

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records).set_index("deal_id")
    
    def compute_pipeline_fee_at_risk(
        self,
        deal_ids: Optional[List[str]] = None,
        loss_df: Optional[pd.DataFrame] = None,
        loss_threshold_ratio: float = 0.02,
        fee_haircut_if_loss: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """Estimate fee-at-risk for deals in the pipeline."""
        if loss_df is None:
            raise ValueError("loss_df (simulation results) must be provided")

        if deal_ids is None:
            deal_ids = list(self.deals.keys())

        sim_index = loss_df.index
        fee_results: Dict[str, pd.DataFrame] = {}

        for deal_id in deal_ids:
            deal = self.deals.get(deal_id)
            if deal is None:
                continue

            deal_tranche_ids = [t.id for t in deal.tranches if t.id in loss_df.columns]
            if not deal_tranche_ids:
                continue

            deal_losses = loss_df[deal_tranche_ids].sum(axis=1)
            total_notional = sum(self.tranches[t].notional for t in deal_tranche_ids)
            total_notional = max(total_notional, 1e-6)
            loss_ratio = deal_losses / total_notional

            base_fee = deal.gross_fees
            haircut_flags = (loss_ratio > loss_threshold_ratio).astype(float)
            bank_fee_outcomes: Dict[str, pd.Series] = {}

            for bank_id, share in deal.bank_share.items():
                bank_base_fee = base_fee * share
                bank_fee_after = bank_base_fee * (1.0 - fee_haircut_if_loss * haircut_flags)
                bank_fee_outcomes[bank_id] = bank_fee_after

            fee_results[deal_id] = pd.DataFrame(bank_fee_outcomes, index=sim_index)

        return fee_results
    
    def aggregate_fee_at_risk(
        self,
        fee_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Aggregate fee-at-risk across deals to get per-bank fee distributions."""
        if not fee_results:
            return pd.DataFrame()

        bank_fee_series: Dict[Tuple[str, str], pd.Series] = {}
        for deal_id, df in fee_results.items():
            for bank_id in df.columns:
                key = (deal_id, bank_id)
                bank_fee_series[key] = df[bank_id]

        panel = pd.DataFrame(bank_fee_series)

        per_bank: Dict[str, pd.Series] = {}
        for (deal_id, bank_id) in panel.columns:
            series = panel[(deal_id, bank_id)]
            if bank_id not in per_bank:
                per_bank[bank_id] = series.copy()
            else:
                per_bank[bank_id] = per_bank[bank_id] + series

        return pd.DataFrame(per_bank)
    
    def monte_carlo_simulation(
        self,
        n_simulations: int = 1000,
        shock_distribution: str = "normal",
        shock_params: Optional[Dict[str, float]] = None,
        diffusion_rate: float = 0.1,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulations to assess risk under various network shocks."""
        entity_ids = list(self.entities.keys())
        n_entities = len(entity_ids)
        
        if shock_params is None:
            shock_params = {"mean": 0.1, "std": 0.05} if shock_distribution == "normal" else {}
        
        final_risks = np.zeros((n_simulations, n_entities))
        
        for sim in range(n_simulations):
            initial_shock: Dict[str, float] = {}
            for entity_id in entity_ids:
                if shock_distribution == "normal":
                    shock = np.random.normal(shock_params.get("mean", 0.1), 
                                            shock_params.get("std", 0.05))
                elif shock_distribution == "uniform":
                    shock = np.random.uniform(shock_params.get("low", 0.0),
                                             shock_params.get("high", 0.2))
                elif shock_distribution == "exponential":
                    shock = np.random.exponential(shock_params.get("scale", 0.1))
                else:
                    shock = 0.1
                
                initial_shock[entity_id] = float(np.clip(shock, 0, 1))
            
            risk_evolution = self.propagate_risk_fluid(
                initial_shock=initial_shock,
                diffusion_rate=diffusion_rate,
                iterations=iterations
            )
            
            final_risks[sim, :] = risk_evolution.iloc[:, -1].values
        
        results: Dict[str, Any] = {
            "mean_risk": pd.Series(np.mean(final_risks, axis=0), index=entity_ids),
            "std_risk": pd.Series(np.std(final_risks, axis=0), index=entity_ids),
            "p5_risk": pd.Series(np.percentile(final_risks, 5, axis=0), index=entity_ids),
            "p95_risk": pd.Series(np.percentile(final_risks, 95, axis=0), index=entity_ids),
            "max_risk": pd.Series(np.max(final_risks, axis=0), index=entity_ids),
            "all_simulations": pd.DataFrame(final_risks, columns=entity_ids),
            "n_simulations": n_simulations
        }
        
        self.simulation_history.append(results)
        return results
    
    def identify_systemic_risk_hubs(self, threshold: float = 0.7) -> List[str]:
        """Identify entities that act as systemic risk hubs (highly connected, high risk)."""
        adj_matrix = self.get_network_adjacency()
        connectivity = adj_matrix.sum(axis=0) + adj_matrix.sum(axis=1)
        risk_levels = pd.Series({
            entity_id: entity.initial_risk_score 
            for entity_id, entity in self.entities.items()
        })
        
        hubs = []
        for entity_id in self.entities.keys():
            conn_score = connectivity[entity_id]
            risk_score = risk_levels[entity_id]
            hub_score = (conn_score / connectivity.max()) * risk_score if connectivity.max() > 0 else 0
            
            if hub_score >= threshold:
                hubs.append(entity_id)
        
        return hubs
    
    def get_risk_contagion_paths(
        self,
        source_entity_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """Find all paths through which risk can propagate from a source entity."""
        if source_entity_id not in self.entities:
            raise ValueError(f"Entity {source_entity_id} not found")
        
        paths: List[List[str]] = []
        
        def dfs(current_id: str, path: List[str], depth: int):
            if depth >= max_depth:
                return
            
            for rel in self.relationships:
                if rel.source_id == current_id and rel.target_id not in path:
                    new_path = path + [rel.target_id]
                    paths.append(new_path)
                    dfs(rel.target_id, new_path, depth + 1)
        
        dfs(source_entity_id, [source_entity_id], 0)
        return paths
    
    def add_markov_chain(self, name: str, chain: MarkovChain) -> None:
        """Add a Markov chain model to Varda."""
        self.markov_chains[name] = chain
    
    def summary(self) -> str:
        """Generate a summary of the Varda instance."""
        n_entities = len(self.entities)
        n_relationships = len(self.relationships)
        n_simulations = len(self.simulation_history)
        n_chains = len(self.markov_chains)
        n_deals = len(self.deals)
        n_tranches = len(self.tranches)
        
        entity_types: Dict[str, int] = {}
        for entity in self.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        deal_types: Dict[str, int] = {}
        for deal in self.deals.values():
            deal_type_name = deal.deal_type.value
            deal_types[deal_type_name] = deal_types.get(deal_type_name, 0) + 1
        
        summary = f"""
Varda Capital Markets Risk Lab: {self.name}
==========================================
Entities: {n_entities}
  {', '.join(f'{k}: {v}' for k, v in entity_types.items())}
Relationships: {n_relationships}
Deals: {n_deals}
  {', '.join(f'{k}: {v}' for k, v in deal_types.items()) if deal_types else 'None'}
Tranches: {n_tranches}
Markov Chains: {n_chains}
Simulations Run: {n_simulations}

Systemic Risk Hubs: {len(self.identify_systemic_risk_hubs())}
        """
        return summary.strip()


# ============================================================================
# VALIDATION FUNCTIONS WITH WEIGHTED PROBABILITIES
# ============================================================================

def validate_weighted_probabilities(
    probabilities: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    tolerance: float = 1e-6
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that weighted probabilities sum to 1.0."""
    if weights is None:
        weights = {k: 1.0 for k in probabilities.keys()}
    
    weighted_sum = sum(p * weights.get(k, 1.0) for k, p in probabilities.items())
    unweighted_sum = sum(probabilities.values())
    negative_probs = {k: p for k, p in probabilities.items() if p < 0}
    
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
    """Validate expected value calculation: E[X] = Σ(x_i * p_i)."""
    analytical_expected = np.sum(values * probabilities)
    analytical_variance = np.sum(values**2 * probabilities) - analytical_expected**2
    prob_sum = np.sum(probabilities)
    prob_sum_valid = abs(prob_sum - 1.0) < tolerance
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
    """Validate that Monte Carlo simulation converges to analytical expected value."""
    n = len(simulated_values)
    sample_mean = np.mean(simulated_values)
    sample_std = np.std(simulated_values, ddof=1)
    
    std_to_use = analytical_std if analytical_std is not None else sample_std
    se = std_to_use / np.sqrt(n)
    
    if stats is not None:
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
    else:
        z_score = 1.96 if confidence_level == 0.95 else 2.576
    
    margin = z_score * se
    ci_lower = analytical_expected - margin
    ci_upper = analytical_expected + margin
    
    is_valid = ci_lower <= sample_mean <= ci_upper
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
    """Validate loss distribution calculations using analytical formulas."""
    pd_annual_adjusted = tranche.pd_annual * scenario.pd_multiplier
    pd_annual_adjusted = min(max(pd_annual_adjusted, 0.0), 1.0)
    pd_horizon = 1.0 - (1.0 - pd_annual_adjusted) ** scenario.horizon_years
    
    discount_factor = (1.0 + risk_free_rate) ** scenario.horizon_years
    analytical_el = pd_horizon * tranche.notional * tranche.lgd / discount_factor
    
    if tranche.id not in loss_df.columns:
        return False, {"error": f"Tranche {tranche.id} not found in loss_df"}
    
    simulated_losses = loss_df[tranche.id].values
    sample_mean = np.mean(simulated_losses)
    sample_std = np.std(simulated_losses, ddof=1)
    
    el_tolerance = analytical_el * 0.05
    el_valid = abs(sample_mean - analytical_el) < el_tolerance
    
    negative_losses = np.sum(simulated_losses < 0)
    non_negative = negative_losses == 0
    
    max_possible_loss = tranche.notional * tranche.lgd / discount_factor
    excessive_losses = np.sum(simulated_losses > max_possible_loss * 1.01)
    within_bounds = excessive_losses == 0
    
    default_rate = np.mean(simulated_losses > 0)
    pd_tolerance = 0.02
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
    """Validate risk propagation calculations."""
    all_risks = risk_evolution.values.flatten()
    in_range = np.all((all_risks >= 0) & (all_risks <= 1))
    
    initial_risks = {eid: e.initial_risk_score for eid, e in varda.entities.items()}
    max_increases = {}
    
    for entity_id in varda.entities.keys():
        if entity_id in risk_evolution.index:
            initial_risk = initial_risks.get(entity_id, 0.0)
            initial_shock_risk = initial_shock.get(entity_id, initial_risk)
            max_risk = risk_evolution.loc[entity_id].max()
            max_increases[entity_id] = max_risk - initial_shock_risk
    
    total_risk_per_iteration = risk_evolution.sum(axis=0)
    risk_growth = total_risk_per_iteration.iloc[-1] - total_risk_per_iteration.iloc[0]
    reasonable_growth = risk_growth < len(varda.entities) * 0.5
    
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


# ============================================================================
# USE CASE DEMONSTRATION: INSTITUTIONS TOKENIZING
# ============================================================================

def create_tokenized_institutions_network() -> Varda:
    """Create a Varda instance with tokenized financial institutions."""
    varda = Varda("Institutions Tokenizing Network")
    
    print("Step 1: Tokenizing major banks...")
    banks = [
        Entity("bank_gs", "Goldman Sachs", "bank", initial_risk_score=0.12,
               metadata={"tier": "GSIB", "region": "US", "assets_usd_bn": 1500, "capital_ratio": 0.15}),
        Entity("bank_jpm", "JPMorgan Chase", "bank", initial_risk_score=0.10,
               metadata={"tier": "GSIB", "region": "US", "assets_usd_bn": 3800, "capital_ratio": 0.16}),
        Entity("bank_ms", "Morgan Stanley", "bank", initial_risk_score=0.14,
               metadata={"tier": "GSIB", "region": "US", "assets_usd_bn": 1200, "capital_ratio": 0.14}),
        Entity("bank_cs", "Credit Suisse", "bank", initial_risk_score=0.25,
               metadata={"tier": "GSIB", "region": "Europe", "assets_usd_bn": 600, "capital_ratio": 0.12}),
        Entity("bank_db", "Deutsche Bank", "bank", initial_risk_score=0.20,
               metadata={"tier": "GSIB", "region": "Europe", "assets_usd_bn": 1400, "capital_ratio": 0.13}),
    ]
    
    initial_ratings = {"bank_gs": "AA-", "bank_jpm": "AA-", "bank_ms": "A+", "bank_cs": "BBB", "bank_db": "BBB+"}
    for bank in banks:
        varda.add_entity(bank, initial_state=initial_ratings.get(bank.id, "A"))
    
    print("Step 2: Tokenizing corporate issuers...")
    issuers = [
        Entity("issuer_techcorp", "TechCorp Inc", "issuer", initial_risk_score=0.15,
               metadata={"sector": "Technology", "revenue_usd_bn": 50, "leverage_ratio": 2.5}),
        Entity("issuer_retailco", "RetailCo Holdings", "issuer", initial_risk_score=0.22,
               metadata={"sector": "Retail", "revenue_usd_bn": 30, "leverage_ratio": 4.0}),
        Entity("issuer_energyco", "EnergyCo Ltd", "issuer", initial_risk_score=0.28,
               metadata={"sector": "Energy", "revenue_usd_bn": 80, "leverage_ratio": 5.5}),
    ]
    
    issuer_ratings = {"issuer_techcorp": "A", "issuer_retailco": "BB+", "issuer_energyco": "BB"}
    for issuer in issuers:
        varda.add_entity(issuer, initial_state=issuer_ratings.get(issuer.id, "BBB"))
    
    print("Step 3: Tokenizing investors and asset managers...")
    investors = [
        Entity("investor_blackrock", "BlackRock", "investor", initial_risk_score=0.08,
               metadata={"type": "asset_manager", "aum_usd_bn": 10000, "strategy": "passive_active_mix"}),
        Entity("investor_vanguard", "Vanguard", "investor", initial_risk_score=0.07,
               metadata={"type": "asset_manager", "aum_usd_bn": 8000, "strategy": "passive"}),
        Entity("investor_pension", "State Pension Fund", "investor", initial_risk_score=0.10,
               metadata={"type": "pension_fund", "aum_usd_bn": 500, "liability_driven": True}),
    ]
    
    for investor in investors:
        varda.add_entity(investor, initial_state="AAA")
    
    print("Step 4: Modeling relationships between institutions...")
    relationships = [
        Relationship("bank_jpm", "bank_gs", "interbank_exposure", 0.15,
                    metadata={"exposure_type": "derivatives", "notional_usd_bn": 5.0}),
        Relationship("bank_gs", "bank_ms", "interbank_exposure", 0.12,
                    metadata={"exposure_type": "repo", "notional_usd_bn": 3.0}),
        Relationship("bank_cs", "bank_db", "interbank_exposure", 0.20,
                    metadata={"exposure_type": "derivatives", "notional_usd_bn": 8.0}),
        Relationship("bank_gs", "issuer_techcorp", "underwriting", 0.25,
                    metadata={"deal_type": "DCM", "exposure_usd_bn": 2.0}),
        Relationship("bank_jpm", "issuer_retailco", "underwriting", 0.30,
                    metadata={"deal_type": "LEVFIN", "exposure_usd_bn": 1.5}),
        Relationship("bank_ms", "issuer_energyco", "underwriting", 0.35,
                    metadata={"deal_type": "DCM_HY", "exposure_usd_bn": 3.0}),
        Relationship("bank_cs", "issuer_techcorp", "underwriting", 0.20,
                    metadata={"deal_type": "ECM", "exposure_usd_bn": 1.0}),
        Relationship("investor_blackrock", "issuer_techcorp", "holdings", 0.40,
                    metadata={"holding_pct": 0.05, "notional_usd_bn": 2.5}),
        Relationship("investor_vanguard", "issuer_retailco", "holdings", 0.35,
                    metadata={"holding_pct": 0.08, "notional_usd_bn": 2.4}),
        Relationship("investor_pension", "issuer_energyco", "holdings", 0.30,
                    metadata={"holding_pct": 0.03, "notional_usd_bn": 2.4}),
        Relationship("bank_gs", "investor_blackrock", "prime_brokerage", 0.18,
                    metadata={"services": ["prime_brokerage", "custody"], "exposure_usd_bn": 10.0}),
        Relationship("bank_jpm", "investor_vanguard", "prime_brokerage", 0.15,
                    metadata={"services": ["custody"], "exposure_usd_bn": 8.0}),
    ]
    
    for rel in relationships:
        varda.add_relationship(rel)
    
    print(f"  Added {len(relationships)} relationships")
    return varda


def add_deals_to_tokenized_network(varda: Varda) -> None:
    """Add capital markets deals that connect tokenized institutions."""
    print("\nStep 5: Adding capital markets deals...")
    
    techcorp_hy_tranche = Tranche(
        id="tranche_techcorp_hy", deal_id="deal_techcorp_hy", currency="USD",
        notional=500_000_000, coupon=0.075, spread_bps=350, maturity_years=7.0,
        rating="BB+", pd_annual=0.025, lgd=0.55, seniority="senior"
    )
    
    techcorp_deal = CapitalMarketsDeal(
        id="deal_techcorp_hy", issuer_entity_id="issuer_techcorp",
        deal_type=DealType.DCM_HY, tranches=[techcorp_hy_tranche],
        bookrunners=["bank_gs", "bank_ms"], co_managers=["bank_jpm"],
        gross_fees=12_500_000, bank_share={"bank_gs": 0.50, "bank_ms": 0.35, "bank_jpm": 0.15},
        pipeline_stage="priced", sector="Technology", region="US"
    )
    
    varda.add_deal(techcorp_deal)
    
    retailco_loan_tranche = Tranche(
        id="tranche_retailco_loan", deal_id="deal_retailco_levfin", currency="USD",
        notional=750_000_000, coupon=0.085, spread_bps=450, maturity_years=6.0,
        rating="B+", pd_annual=0.04, lgd=0.60, seniority="senior", is_secured=True
    )
    
    retailco_deal = CapitalMarketsDeal(
        id="deal_retailco_levfin", issuer_entity_id="issuer_retailco",
        deal_type=DealType.LEVFIN_LBO, tranches=[retailco_loan_tranche],
        bookrunners=["bank_jpm"], gross_fees=22_500_000,
        bank_share={"bank_jpm": 1.0}, pipeline_stage="priced", sector="Retail", region="US"
    )
    
    varda.add_deal(retailco_deal)
    
    energyco_ig_tranche = Tranche(
        id="tranche_energyco_ig", deal_id="deal_energyco_ig", currency="USD",
        notional=1_000_000_000, coupon=0.045, spread_bps=150, maturity_years=10.0,
        rating="BBB", pd_annual=0.015, lgd=0.45, seniority="senior"
    )
    
    energyco_deal = CapitalMarketsDeal(
        id="deal_energyco_ig", issuer_entity_id="issuer_energyco",
        deal_type=DealType.DCM_IG, tranches=[energyco_ig_tranche],
        bookrunners=["bank_ms", "bank_cs"], gross_fees=15_000_000,
        bank_share={"bank_ms": 0.60, "bank_cs": 0.40},
        pipeline_stage="priced", sector="Energy", region="US"
    )
    
    varda.add_deal(energyco_deal)
    print(f"  Added {len(varda.deals)} deals with {len(varda.tranches)} tranches")


def run_validation_suite(varda: Varda, loss_df: Optional[pd.DataFrame] = None, scenario: Optional[CapitalMarketsScenario] = None) -> Dict[str, Any]:
    """Run comprehensive validation suite with weighted probabilities."""
    print("\n" + "="*70)
    print("VALIDATION SUITE: WEIGHTED PROBABILITY VERIFICATION")
    print("="*70)
    
    validation_results = {}
    
    print("\n1. Validating Markov Chain Probabilities...")
    if "market_regimes" in varda.markov_chains:
        chain = varda.markov_chains["market_regimes"]
        stationary = chain.stationary_distribution()
        probs = {state: float(p) for state, p in zip(chain.states, stationary)}
        is_valid, report = validate_weighted_probabilities(probs)
        validation_results["markov_chain"] = {"valid": is_valid, "report": report}
        
        print(f"   Stationary distribution probabilities:")
        for state, prob in probs.items():
            print(f"     {state}: {prob:.6f}")
        print(f"   Sum: {sum(probs.values()):.6f} (should be 1.0)")
        print(f"   Valid: {is_valid} ✓" if is_valid else f"   Valid: {is_valid} ✗")
    
    if loss_df is not None and scenario is not None:
        print("\n2. Validating Loss Distribution Calculations...")
        loss_validations = {}
        for tranche_id, tranche in varda.tranches.items():
            if tranche_id in loss_df.columns:
                is_valid, report = validate_loss_distribution(tranche, scenario, loss_df)
                loss_validations[tranche_id] = {"valid": is_valid, "report": report}
                print(f"   {tranche_id}: EL_diff=${report['el_difference']:,.2f}, Valid={is_valid} ✓" if is_valid else f"   {tranche_id}: Valid={is_valid} ✗")
        validation_results["loss_distributions"] = loss_validations
        
        print("\n3. Validating Monte Carlo Convergence...")
        mc_validations = {}
        for tranche_id, tranche in varda.tranches.items():
            if tranche_id in loss_df.columns:
                simulated_losses = loss_df[tranche_id].values
                pd_horizon = 1.0 - (1.0 - tranche.pd_annual * scenario.pd_multiplier) ** scenario.horizon_years
                discount_factor = 1.03
                analytical_el = pd_horizon * tranche.notional * tranche.lgd / discount_factor
                analytical_std = np.sqrt(pd_horizon * (1 - pd_horizon) * (tranche.notional * tranche.lgd / discount_factor) ** 2)
                
                is_valid, report = validate_monte_carlo_convergence(simulated_losses, analytical_el, analytical_std)
                mc_validations[tranche_id] = {"valid": is_valid, "report": report}
                print(f"   {tranche_id}: Within CI={report['within_ci']}, Rel Error={report['relative_error']*100:.2f}%")
        validation_results["monte_carlo"] = mc_validations
    
    print("\n4. Validating Relationship Weights...")
    relationship_weights = {f"{rel.source_id}->{rel.target_id}": rel.strength for rel in varda.relationships}
    all_valid = all(0 <= w <= 1 for w in relationship_weights.values())
    validation_results["relationships"] = {"valid": all_valid, "num_relationships": len(relationship_weights)}
    print(f"   Total relationships: {len(relationship_weights)}")
    print(f"   All in [0,1]: {all_valid} ✓" if all_valid else f"   All in [0,1]: {all_valid} ✗")
    
    print("\n5. Validating Entity Risk Scores...")
    risk_scores = {eid: e.initial_risk_score for eid, e in varda.entities.items()}
    all_risks_valid = all(0 <= r <= 1 for r in risk_scores.values())
    validation_results["risk_scores"] = {"valid": all_risks_valid, "num_entities": len(risk_scores)}
    print(f"   Total entities: {len(risk_scores)}")
    print(f"   All in [0,1]: {all_risks_valid} ✓" if all_risks_valid else f"   All in [0,1]: {all_risks_valid} ✗")
    
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


def main():
    """Main demonstration: Tokenizing institutions and analyzing their network."""
    print("="*70)
    print("VARDA: INSTITUTIONS TOKENIZING USE CASE DEMONSTRATION")
    print("="*70)
    print("\nThis demonstrates how Varda can be used to tokenize")
    print("financial institutions and analyze their network relationships.\n")
    
    varda = create_tokenized_institutions_network()
    add_deals_to_tokenized_network(varda)
    
    print("\n" + "="*70)
    print("NETWORK ANALYSIS")
    print("="*70)
    adj_matrix = varda.get_network_adjacency()
    print(f"Total entities: {len(varda.entities)}")
    print(f"Total relationships: {len(varda.relationships)}")
    
    hubs = varda.identify_systemic_risk_hubs(threshold=0.15)
    print(f"\nSystemic Risk Hubs: {len(hubs)}")
    for hub_id in hubs[:3]:
        entity = varda.entities[hub_id]
        print(f"  - {entity.name} (Risk: {entity.initial_risk_score:.3f})")
    
    print("\n" + "="*70)
    print("RISK PROPAGATION SIMULATION")
    print("="*70)
    initial_shock = {"bank_cs": 0.6}
    risk_evolution = varda.propagate_risk_fluid(initial_shock=initial_shock, diffusion_rate=0.15, iterations=10)
    final_risks = risk_evolution.iloc[:, -1].sort_values(ascending=False)
    print("\nTop 5 Most Affected Institutions:")
    for entity_id, risk in final_risks.head(5).items():
        entity = varda.entities[entity_id]
        change = risk - entity.initial_risk_score
        print(f"  {entity.name:30s} Risk: {risk:.3f} (change: {change:+.3f})")
    
    is_valid, _ = validate_risk_propagation(varda, initial_shock, risk_evolution, 0.15)
    print(f"\nRisk Propagation Valid: {is_valid} ✓" if is_valid else f"\nRisk Propagation Valid: {is_valid} ✗")
    
    print("\n" + "="*70)
    print("DEAL-LEVEL RISK ANALYSIS")
    print("="*70)
    
    market_chain = create_market_regime_chain()
    varda.add_markov_chain("market_regimes", market_chain)
    
    stress_constraint = MarketConstraint(
        name="Credit Spread Widening", constraint_type="market", value=200.0,
        impact_on_transitions={"Normal->Stressed": 1.8, "Stressed->Crisis": 2.0}
    )
    
    pd_multiplier = varda.calibrate_pd_multiplier_from_regime(
        "market_regimes", [stress_constraint], "Normal", "Crisis"
    )
    print(f"\nPD Multiplier from stress scenario: {pd_multiplier:.2f}x")
    
    stress_scenario = CapitalMarketsScenario(
        name="Credit Stress Scenario", description="200bps spread widening, elevated default rates",
        spread_shock_bps=200.0, pd_multiplier=pd_multiplier, horizon_years=1.0,
        market_constraints=[stress_constraint]
    )
    
    print("\nRunning loss distribution simulation...")
    loss_df = varda.simulate_tranche_loss_distribution(
        tranche_ids=list(varda.tranches.keys()), scenario=stress_scenario,
        n_simulations=10000, random_seed=42
    )
    
    loss_summary = varda.summarize_loss_distribution(loss_df, var_levels=[0.95, 0.99])
    print("\nTranche Loss Summary:")
    print(loss_summary.to_string())
    
    pipeline_summary = varda.summarize_pipeline_risk_and_return(None, loss_df, 0.99)
    print("\nDeal-Level Risk/Return Summary:")
    print(pipeline_summary.to_string())
    
    validation_results = run_validation_suite(varda, loss_df, stress_scenario)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(varda.summary())
    
    if validation_results.get("overall_valid", False):
        print("\n✓ All validations passed - methodology verified with weighted probabilities")
    else:
        print("\n✗ Some validations failed - review validation results above")


if __name__ == "__main__":
    main()

