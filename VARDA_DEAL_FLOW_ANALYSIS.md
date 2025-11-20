# Varda in Capital Markets Deal Flow: Token Analysis & Risk Assessment

## Executive Summary

Varda provides risk analytics and network analysis capabilities throughout the entire capital markets deal lifecycle. This document explains where Varda fits in each stage of a transaction and what equations/formulas are used to analyze tokenized institutions.

---

## Deal Flow Stages & Varda's Role

### Stage 1: IDEA / PIPELINE BUILDING
**Deal Status:** `pipeline_stage = "idea"`

**What Varda Does:**
- **Network Analysis**: Identify which institutions (tokens) are connected to potential issuers
- **Systemic Risk Assessment**: Evaluate if adding this deal increases systemic risk
- **Capacity Analysis**: Check if banks have capacity for new exposures
- **Competitive Intelligence**: Analyze competitor relationships with issuer

**Key Equations:**
```python
# 1. Network Centrality Score (identify key institutions)
centrality_score = (in_degree + out_degree) / max_possible_connections

# 2. Systemic Risk Contribution
systemic_risk_contribution = risk_score × connectivity × exposure_size

# 3. Capacity Utilization
capacity_ratio = current_exposures / regulatory_capital_limit
```

**Varda Functions:**
```python
# Identify systemic hubs that might be affected
hubs = varda.identify_systemic_risk_hubs(threshold=0.7)

# Analyze network topology
adj_matrix = varda.get_network_adjacency()
network_density = adj_matrix.sum().sum() / (n_entities * (n_entities - 1))

# Check existing relationships
issuer_relationships = [r for r in varda.relationships 
                        if r.target_id == issuer_id or r.source_id == issuer_id]
```

---

### Stage 2: MANDATED / ENGAGEMENT
**Deal Status:** `pipeline_stage = "mandated"`

**What Varda Does:**
- **Counterparty Risk Assessment**: Evaluate issuer's credit risk and network position
- **Underwriting Capacity**: Model how this deal affects bank's risk profile
- **Fee-at-Risk Analysis**: Estimate probability of fee impairment
- **Regulatory Capital Impact**: Project capital consumption

**Key Equations:**
```python
# 1. Expected Loss (EL) for Underwriting Exposure
EL = PD_horizon × Notional × LGD × Discount_Factor

# Where:
PD_horizon = 1 - (1 - PD_annual)^horizon_years
Discount_Factor = 1 / (1 + risk_free_rate)^horizon_years

# 2. Value-at-Risk (VaR) at confidence level α
VaR_α = quantile(loss_distribution, α)

# 3. Expected Shortfall (ES) / Conditional VaR
ES_α = E[Loss | Loss ≥ VaR_α]

# 4. Fee-at-Risk (probability-weighted fee impairment)
Fee_at_Risk = E[Fee × (1 - haircut_flag)]
haircut_flag = 1 if (loss / notional) > threshold else 0
```

**Varda Functions:**
```python
# Create deal in pipeline
deal = CapitalMarketsDeal(
    id="deal_new",
    issuer_entity_id="issuer_techcorp",
    deal_type=DealType.DCM_HY,
    pipeline_stage="mandated",  # ← At this stage
    ...
)
varda.add_deal(deal)

# Run preliminary risk assessment
preliminary_scenario = CapitalMarketsScenario(
    name="Baseline Assessment",
    pd_multiplier=1.0,  # No stress yet
    horizon_years=1.0
)

# Estimate expected loss (even before pricing)
estimated_el = tranche.pd_annual * tranche.notional * tranche.lgd
```

---

### Stage 3: LAUNCHED / MARKETING
**Deal Status:** `pipeline_stage = "launched"`

**What Varda Does:**
- **Market Regime Analysis**: Assess current market conditions and adjust PDs
- **Investor Network Analysis**: Identify which investors (tokens) are likely buyers
- **Syndicate Risk Sharing**: Model risk distribution across bookrunners
- **Pricing Sensitivity**: Analyze how pricing affects risk-return profile

**Key Equations:**
```python
# 1. Regime-Adjusted PD Multiplier
PD_multiplier = f(market_regime_state, economic_indicators)

# From Markov chain steady state:
PD_multiplier = P(Crisis | constraints) / P(Crisis | baseline)

# 2. Risk-Adjusted Return on Capital (RAROC)
RAROC = (Fee_Income - Expected_Loss) / Economic_Capital

# 3. Sharpe-like Ratio for Deals
Deal_Sharpe = (Fee - EL) / VaR_99

# 4. Network Contagion Probability
contagion_prob = P(issuer_default | connected_entity_default)
```

**Varda Functions:**
```python
# Analyze market regime
market_chain = create_market_regime_chain()
varda.add_markov_chain("market_regimes", market_chain)

# Get current regime probabilities
steady_state = varda.analyze_market_steady_state("market_regimes")
crisis_prob = steady_state["steady_state"]["Crisis"]

# Calibrate PD multiplier from regime
pd_mult = varda.calibrate_pd_multiplier_from_regime(
    "market_regimes",
    scenario_constraints=[current_market_constraints],
    base_state="Normal",
    stressed_state="Crisis"
)

# Analyze investor network
investor_connections = [r for r in varda.relationships 
                       if r.source_id.startswith("investor_") 
                       and r.target_id == issuer_id]

# Model syndicate risk sharing
for bank_id in deal.bookrunners:
    bank_exposure = deal.gross_fees * deal.bank_share[bank_id]
    bank_risk = varda.entities[bank_id].initial_risk_score
    # Risk concentration analysis
```

---

### Stage 4: PRICED / EXECUTION
**Deal Status:** `pipeline_stage = "priced"`

**What Varda Does:**
- **Final Risk Quantification**: Run full Monte Carlo loss distribution
- **Capital Allocation**: Determine economic capital required
- **Stress Testing**: Test deal under adverse scenarios
- **Fee Validation**: Verify fees compensate for tail risk

**Key Equations:**
```python
# 1. Full Loss Distribution (Monte Carlo)
For each simulation i:
    default_flag_i ~ Bernoulli(PD_horizon)
    loss_i = default_flag_i × Notional × LGD / (1 + r)^horizon

# 2. Economic Capital (EC)
EC = VaR_99.9 - EL  # or ES_99.9 - EL for regulatory

# 3. Risk-Adjusted Pricing Check
if Fee < (EL + λ × EC):
    pricing_adequate = False
else:
    pricing_adequate = True

# Where λ is risk appetite parameter (typically 0.1-0.3)

# 4. Portfolio Risk Contribution (if deal added to existing portfolio)
Marginal_VaR = VaR(portfolio + deal) - VaR(portfolio)
Incremental_VaR = Marginal_VaR / deal_notional
```

**Varda Functions:**
```python
# Update deal status
deal.pipeline_stage = "priced"

# Run full loss distribution simulation
stress_scenario = CapitalMarketsScenario(
    name="Pricing Validation",
    pd_multiplier=pd_mult,
    spread_shock_bps=0,  # Actual pricing
    horizon_years=1.0
)

loss_df = varda.simulate_tranche_loss_distribution(
    tranche_ids=[tranche.id],
    scenario=stress_scenario,
    n_simulations=10000,
    random_seed=42
)

# Calculate risk metrics
loss_summary = varda.summarize_loss_distribution(loss_df, var_levels=[0.95, 0.99, 0.999])

# Deal-level risk-return analysis
deal_summary = varda.summarize_deal_risk_and_return(deal.id, loss_df, var_level=0.99)

# Check if fees compensate for risk
if deal_summary["VaR_over_fees"] > 2.0:  # VaR > 2x fees
    print("WARNING: Tail risk exceeds fee compensation")

# Fee-at-risk analysis
fee_at_risk = varda.compute_pipeline_fee_at_risk(
    deal_ids=[deal.id],
    loss_df=loss_df,
    loss_threshold_ratio=0.02,
    fee_haircut_if_loss=0.5
)
```

---

### Stage 5: CLOSED / ON-BOOK
**Deal Status:** `pipeline_stage = "closed"`

**What Varda Does:**
- **Ongoing Monitoring**: Track issuer credit quality and network position
- **Mark-to-Market Risk**: Model valuation changes
- **Refinancing Risk**: Analyze refi-wall and rollover risk
- **Portfolio Aggregation**: Add to bank's total exposure

**Key Equations:**
```python
# 1. Credit Migration Risk (Markov Chain)
P(rating_t+1 | rating_t) = transition_matrix[rating_t, rating_t+1]

# 2. Default Probability Evolution
PD_t = P(Default | current_rating, time_to_maturity)

# 3. Mark-to-Market Loss (for traded instruments)
MTM_Loss = (Current_Spread - Initial_Spread) × DV01 × Notional

# 4. Refinancing Risk
Refi_Risk = P(unable_to_refinance | market_conditions, issuer_rating)

# 5. Portfolio Concentration Risk
HHI = Σ(Exposure_i / Total_Exposure)^2  # Herfindahl-Hirschman Index
Concentration_Risk = HHI × Average_PD
```

**Varda Functions:**
```python
# Update deal status
deal.pipeline_stage = "closed"

# Ongoing monitoring: simulate rating transitions
rating_chain = create_credit_rating_chain()
varda.add_markov_chain("credit_ratings", rating_chain)

# Simulate issuer rating evolution
rating_evolution = varda.simulate_entity_transitions(
    chain_name="credit_ratings",
    entity_ids=[deal.issuer_entity_id],
    n_steps=12  # Monthly for 1 year
)

# Compute default probability over remaining maturity
default_probs = varda.compute_default_probabilities(
    chain_name="credit_ratings",
    horizon=remaining_months,
    entity_ids=[deal.issuer_entity_id]
)

# Network monitoring: check if issuer becomes systemic risk hub
if deal.issuer_entity_id in varda.identify_systemic_risk_hubs(threshold=0.6):
    print("WARNING: Issuer has become a systemic risk hub")

# Portfolio aggregation
total_pipeline_risk = varda.summarize_pipeline_risk_and_return(
    deal_ids=None,  # All deals
    loss_df=combined_loss_df,
    var_level=0.99
)
```

---

## Key Equations for Token (Institution) Analysis

### 1. Risk Score Calculation
```python
# Base risk score (0-1 scale)
risk_score = f(capital_ratio, leverage, rating, market_conditions)

# Example formula:
risk_score = (
    0.4 × (1 - capital_ratio) +           # Capital adequacy
    0.3 × min(leverage_ratio / 10, 1) +   # Leverage (capped)
    0.2 × rating_to_risk(rating) +         # Credit rating
    0.1 × market_stress_factor             # Market conditions
)
```

### 2. Network Risk Propagation
```python
# Fluid dynamics-inspired diffusion
risk(t+1) = (1 - α) × risk(t) + α × diffused_risk(t)

# Where:
diffused_risk(t) = normalized_adjacency_matrix × risk(t)
α = diffusion_rate (typically 0.1-0.2)

# Normalized adjacency:
normalized_adj[i,j] = relationship_strength[i,j] / Σ_k relationship_strength[i,k]
```

### 3. Systemic Risk Hub Identification
```python
# Hub score combines connectivity and risk
hub_score = (connectivity_normalized × risk_score)

# Where:
connectivity = in_degree + out_degree
connectivity_normalized = connectivity / max_connectivity_in_network

# Entity is a hub if:
hub_score ≥ threshold  # typically 0.6-0.8
```

### 4. Contagion Path Analysis
```python
# Find all paths from source to target
paths = find_all_paths(source_entity, target_entity, max_depth)

# Contagion probability along path
P(contagion | path) = Π(relationship_strength[i] along path)

# Total contagion probability
P(contagion) = 1 - Π(1 - P(contagion | path_i) for all paths)
```

### 5. Expected Loss with Network Effects
```python
# Base EL (standalone)
EL_standalone = PD × Notional × LGD

# Network-adjusted EL
EL_network = EL_standalone × (1 + network_risk_multiplier)

# Where network_risk_multiplier accounts for:
# - Contagion from connected entities
# - Systemic risk contribution
# - Concentration risk
network_risk_multiplier = f(
    connected_entity_risks,
    relationship_strengths,
    systemic_risk_score
)
```

### 6. Portfolio-Level Risk Aggregation
```python
# Simple sum (assuming independence - conservative)
Portfolio_VaR_independent = Σ VaR_i

# With correlation (more realistic)
Portfolio_VaR_correlated = √(Σ Σ VaR_i × correlation[i,j] × VaR_j)

# Network-adjusted (accounts for contagion)
Portfolio_VaR_network = Portfolio_VaR_correlated × network_amplification_factor

# Where network_amplification_factor > 1 captures:
# - Contagion effects
# - Common exposures
# - Systemic risk
```

---

## Practical Workflow Example

### Scenario: New High-Yield Bond Deal

```python
# STAGE 1: IDEA
# Analyze issuer network position
issuer = varda.entities["issuer_techcorp"]
issuer_risk = issuer.initial_risk_score
issuer_connections = count_relationships(issuer.id)

# Check if issuer is systemic risk hub
if issuer.id in varda.identify_systemic_risk_hubs():
    print("CAUTION: Issuer is a systemic risk hub")

# STAGE 2: MANDATED
# Create deal in pipeline
deal = CapitalMarketsDeal(
    id="deal_new_hy",
    issuer_entity_id="issuer_techcorp",
    deal_type=DealType.DCM_HY,
    pipeline_stage="mandated",
    tranches=[tranche],
    bookrunners=["bank_gs", "bank_jpm"],
    gross_fees=10_000_000
)
varda.add_deal(deal)

# Preliminary risk assessment
prelim_el = tranche.pd_annual * tranche.notional * tranche.lgd
print(f"Preliminary EL: ${prelim_el:,.0f}")

# STAGE 3: LAUNCHED
# Market regime analysis
pd_mult = varda.calibrate_pd_multiplier_from_regime(
    "market_regimes",
    scenario_constraints=[current_constraints],
    base_state="Normal",
    stressed_state="Crisis"
)
print(f"PD multiplier from market regime: {pd_mult:.2f}x")

# STAGE 4: PRICED
deal.pipeline_stage = "priced"

# Full risk quantification
scenario = CapitalMarketsScenario(
    name="Pricing Validation",
    pd_multiplier=pd_mult,
    horizon_years=1.0
)

loss_df = varda.simulate_tranche_loss_distribution(
    tranche_ids=[tranche.id],
    scenario=scenario,
    n_simulations=10000
)

# Risk metrics
summary = varda.summarize_loss_distribution(loss_df)
el = summary.loc[tranche.id, "EL"]
var_99 = summary.loc[tranche.id, "VaR_99"]

# Risk-return check
deal_summary = varda.summarize_deal_risk_and_return(deal.id, loss_df)
if deal_summary["VaR_over_fees"] > 2.0:
    print("WARNING: Tail risk high relative to fees")

# STAGE 5: CLOSED
deal.pipeline_stage = "closed"

# Ongoing monitoring
rating_evolution = varda.simulate_entity_transitions(
    "credit_ratings",
    entity_ids=[issuer.id],
    n_steps=12
)

# Portfolio aggregation
all_deals_risk = varda.summarize_pipeline_risk_and_return(
    deal_ids=None,
    loss_df=combined_loss_df
)
```

---

## Summary: Varda's Value Across Deal Flow

| Stage | Key Varda Capabilities | Primary Equations |
|-------|------------------------|-------------------|
| **Idea** | Network analysis, systemic risk identification | Centrality, HHI, connectivity |
| **Mandated** | Counterparty risk, capacity analysis | EL, PD_horizon, capacity ratios |
| **Launched** | Market regime analysis, investor network | PD_multiplier, RAROC, Sharpe |
| **Priced** | Full risk quantification, stress testing | VaR, ES, Fee-at-Risk, EC |
| **Closed** | Ongoing monitoring, portfolio aggregation | Rating transitions, MTM, refi risk |

---

## Key Takeaways

1. **Varda provides continuous risk analytics** throughout the deal lifecycle
2. **Token analysis** (institution analysis) uses network metrics, risk scores, and connectivity measures
3. **Key equations** focus on: Expected Loss, VaR/ES, network propagation, and systemic risk
4. **Validation** ensures calculations are correct using weighted probabilities and analytical checks
5. **Integration** with deal pipeline stages enables real-time risk decision-making

---

*This document demonstrates how Varda can be integrated into capital markets workflows to provide quantitative risk analysis at every stage of a transaction.*

