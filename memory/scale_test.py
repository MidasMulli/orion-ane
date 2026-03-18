#!/usr/bin/env python3
"""
Scale Test for Memory Daemon
=============================
500+ realistic ISDA/finance/banking facts across 6 months of simulated conversations.
Tests ingestion throughput, retrieval quality, contradiction handling, and dedup.

Usage:
    /Users/midas/.mlx-env/bin/python3 /Users/midas/Desktop/cowork/orion-ane/memory/scale_test.py
"""

import os
import sys
import time
import shutil
import random
import tracemalloc
from datetime import datetime, timedelta

# Add parent to path for daemon import
try:
    from phantom_memory.daemon import MemoryDaemon
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from daemon import MemoryDaemon

# ── Test Configuration ────────────────────────────────────────────

VAULT_PATH = "/Users/midas/Desktop/cowork/orion-ane/memory/scale_test_vault"
DB_PATH = "/Users/midas/Desktop/cowork/orion-ane/memory/scale_test_chromadb"

# 20 counterparties with realistic profiles
COUNTERPARTIES = [
    {"name": "Meridian Capital Partners", "rating": "A+", "sector": "Hedge Fund", "jurisdiction": "London", "aum": "$4.2 billion"},
    {"name": "Pacific Rim Securities", "rating": "BBB+", "sector": "Broker-Dealer", "jurisdiction": "Hong Kong", "aum": "$1.8 billion"},
    {"name": "Nordic Sovereign Fund", "rating": "AAA", "sector": "Sovereign Wealth", "jurisdiction": "Oslo", "aum": "$85 billion"},
    {"name": "Apex Trading Group", "rating": "BBB", "sector": "Proprietary Trading", "jurisdiction": "New York", "aum": "$950 million"},
    {"name": "Sterling Asset Management", "rating": "A-", "sector": "Asset Manager", "jurisdiction": "London", "aum": "$12.5 billion"},
    {"name": "Shanghai Dragon Capital", "rating": "BBB-", "sector": "Private Equity", "jurisdiction": "Shanghai", "aum": "$3.1 billion"},
    {"name": "Atlas Global Fund", "rating": "A", "sector": "Multi-Strategy", "jurisdiction": "Geneva", "aum": "$7.6 billion"},
    {"name": "Vanguard Credit Partners", "rating": "BBB+", "sector": "Credit Fund", "jurisdiction": "New York", "aum": "$2.3 billion"},
    {"name": "Sakura Financial Holdings", "rating": "A+", "sector": "Bank", "jurisdiction": "Tokyo", "aum": "$45 billion"},
    {"name": "Condor Macro Fund", "rating": "BB+", "sector": "Macro Fund", "jurisdiction": "Cayman Islands", "aum": "$680 million"},
    {"name": "Rhine Valley Bank", "rating": "AA-", "sector": "Bank", "jurisdiction": "Frankfurt", "aum": "$120 billion"},
    {"name": "Emerald Isle Capital", "rating": "BBB", "sector": "Hedge Fund", "jurisdiction": "Dublin", "aum": "$1.1 billion"},
    {"name": "Falcon Bridge Securities", "rating": "A-", "sector": "Broker-Dealer", "jurisdiction": "Singapore", "aum": "$5.4 billion"},
    {"name": "Pinnacle Quant Fund", "rating": "BBB+", "sector": "Quantitative", "jurisdiction": "Chicago", "aum": "$890 million"},
    {"name": "Aurora Structured Products", "rating": "A", "sector": "Structured Finance", "jurisdiction": "London", "aum": "$3.7 billion"},
    {"name": "Maple Leaf Investments", "rating": "AA", "sector": "Pension Fund", "jurisdiction": "Toronto", "aum": "$28 billion"},
    {"name": "Caspian Energy Fund", "rating": "BB", "sector": "Commodity Fund", "jurisdiction": "Dubai", "aum": "$420 million"},
    {"name": "Delta Prime Capital", "rating": "BBB-", "sector": "Distressed Debt", "jurisdiction": "New York", "aum": "$1.6 billion"},
    {"name": "Oceanic Trading Corp", "rating": "BBB", "sector": "Proprietary Trading", "jurisdiction": "Sydney", "aum": "$750 million"},
    {"name": "Summit Peak Holdings", "rating": "A+", "sector": "Family Office", "jurisdiction": "Zurich", "aum": "$9.2 billion"},
]

# Date range: 2025-09-01 to 2026-03-17
START_DATE = datetime(2025, 9, 1)
END_DATE = datetime(2026, 3, 17)
TOTAL_DAYS = (END_DATE - START_DATE).days


def random_date() -> datetime:
    return START_DATE + timedelta(days=random.randint(0, TOTAL_DAYS))


def date_in_month(year: int, month: int) -> datetime:
    day = random.randint(1, 28)
    return datetime(year, month, day)


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Generate realistic data
# ══════════════════════════════════════════════════════════════════

def generate_counterparty_facts() -> list[dict]:
    """Generate counterparty profile facts."""
    facts = []
    for cp in COUNTERPARTIES:
        dt = random_date()
        # Basic profile
        facts.append({"role": "assistant", "text": f"{cp['name']} is a {cp['rating']} rated {cp['sector']} based in {cp['jurisdiction']} with AUM of {cp['aum']}.", "date": dt})
        facts.append({"role": "user", "text": f"We're onboarding {cp['name']} as a new counterparty. They're a {cp['sector']} out of {cp['jurisdiction']}.", "date": dt})
        # Rating details
        facts.append({"role": "assistant", "text": f"The current credit rating for {cp['name']} is {cp['rating']} from S&P, confirmed in the latest annual review.", "date": dt + timedelta(days=random.randint(1, 5))})
    return facts


def generate_isda_facts() -> list[dict]:
    """Generate ISDA Master Agreement terms for various counterparties."""
    facts = []
    thresholds = ["$25 million", "$50 million", "$75 million", "$100 million", "$150 million"]
    netting_types = [
        "Full close-out netting under Section 6(e) with bilateral payment netting",
        "Payment netting under Section 2(c) for same-currency same-day obligations only",
        "Close-out netting with single agreement approach across all transactions",
        "Selective netting with carve-outs for credit derivative transactions",
    ]
    termination_events = [
        "Automatic Early Termination applies for Counterparty as Defaulting Party",
        "Credit Event Upon Merger is specified as an Additional Termination Event",
        "Force Majeure provisions apply with 3 business day waiting period",
        "NAV trigger at 25% decline as Additional Termination Event",
        "Change of control triggers Additional Termination Event requiring 30 days notice",
    ]

    for cp in COUNTERPARTIES:
        dt = random_date()
        threshold = random.choice(thresholds)
        facts.append({"role": "assistant", "text": f"The cross-default threshold for {cp['name']} is set at {threshold} including affiliate indebtedness under Section 5(a)(vi).", "date": dt})
        facts.append({"role": "assistant", "text": f"Netting provisions for {cp['name']}: {random.choice(netting_types)}.", "date": dt + timedelta(days=1)})
        facts.append({"role": "assistant", "text": f"For {cp['name']}, {random.choice(termination_events)}.", "date": dt + timedelta(days=2)})
        # Credit support
        facts.append({"role": "user", "text": f"The credit support requirements for {cp['name']} include an independent amount of {random.choice(['$5 million', '$10 million', '$15 million', '$20 million'])}.", "date": dt + timedelta(days=3)})
    return facts


def generate_csa_facts() -> list[dict]:
    """Generate CSA (Credit Support Annex) terms."""
    facts = []
    collateral_sets = [
        "cash in USD, EUR, and GBP only",
        "cash in USD plus US Treasuries with 2% haircut",
        "cash in any G7 currency plus G7 government securities with 2% haircut and Agency securities with 4% haircut",
        "cash in USD/EUR plus US Treasuries at 2% and investment-grade corporate bonds at 8% haircut",
        "cash only in USD with no securities accepted",
    ]
    mta_values = ["$250,000", "$500,000", "$1 million", "$100,000"]
    valuation_days = ["every business day", "weekly on Mondays", "bi-weekly", "monthly on the first business day"]

    for cp in COUNTERPARTIES:
        dt = random_date()
        collateral = random.choice(collateral_sets)
        threshold_val = random.choice(["zero", "$5 million", "$10 million", "$25 million"])
        mta = random.choice(mta_values)
        val_day = random.choice(valuation_days)

        facts.append({"role": "assistant", "text": f"CSA for {cp['name']} specifies eligible collateral as {collateral}.", "date": dt})
        facts.append({"role": "assistant", "text": f"The threshold amount for {cp['name']} under the CSA is {threshold_val} with a minimum transfer amount of {mta}.", "date": dt + timedelta(days=1)})
        facts.append({"role": "assistant", "text": f"Valuation dates for the {cp['name']} CSA are {val_day} with notification by 1:00 PM local time.", "date": dt + timedelta(days=2)})
        # Haircut specifics
        facts.append({"role": "user", "text": f"For {cp['name']}, the haircut schedule is: US Treasuries at 2%, Agency securities at 4%, and investment-grade corporates at 8% with maturity-based adjustments.", "date": dt + timedelta(days=3)})
    return facts


def generate_decision_facts() -> list[dict]:
    """Generate decision facts (we decided to, agreed on, etc.)."""
    facts = []
    decisions = [
        ("Meridian Capital Partners", "We decided to increase the cross-default threshold from $50 million to $75 million for Meridian Capital Partners given their strong AUM position."),
        ("Pacific Rim Securities", "We agreed to accept Pacific Rim Securities' request for payment netting across all currency pairs under Section 2(c)."),
        ("Nordic Sovereign Fund", "We decided to offer Nordic Sovereign Fund a zero threshold under the CSA given their AAA sovereign rating."),
        ("Apex Trading Group", "We agreed on a $25 million threshold for Apex Trading Group with quarterly review triggers tied to their NAV."),
        ("Sterling Asset Management", "We decided to go with bilateral close-out netting for Sterling Asset Management with no carve-outs."),
        ("Atlas Global Fund", "We chose to include Automatic Early Termination for Atlas Global Fund as the Defaulting Party only."),
        ("Vanguard Credit Partners", "We settled on eligible collateral of cash plus US Treasuries for Vanguard Credit Partners, rejecting their request for corporate bonds."),
        ("Sakura Financial Holdings", "We agreed to include Force Majeure provisions with a 5 business day waiting period for Sakura Financial Holdings."),
        ("Rhine Valley Bank", "We decided to accept Rhine Valley Bank's standard ISDA terms with minimal modifications given their AA- rating."),
        ("Falcon Bridge Securities", "We agreed on a minimum transfer amount of $500,000 for Falcon Bridge Securities, down from our initial $1 million proposal."),
        ("Summit Peak Holdings", "We confirmed that Summit Peak Holdings will have an independent amount of $15 million under the CSA."),
        ("Delta Prime Capital", "We decided against including credit event upon merger for Delta Prime Capital after legal review."),
        ("Emerald Isle Capital", "We agreed to grant Emerald Isle Capital a 30-day cure period for cross-default events."),
        ("Condor Macro Fund", "We decided to require daily margin calls for Condor Macro Fund given their BB+ rating and macro fund volatility profile."),
        ("Pinnacle Quant Fund", "We settled on a NAV trigger at 20% decline as an Additional Termination Event for Pinnacle Quant Fund."),
    ]
    for i, (cp, text) in enumerate(decisions):
        dt = date_in_month(2025, 10) + timedelta(days=i * 2)
        facts.append({"role": "user", "text": text, "date": dt})
    return facts


def generate_task_facts() -> list[dict]:
    """Generate task/deadline facts."""
    facts = []
    tasks = [
        "We need to review the Meridian Capital Partners ISDA schedule by end of next week.",
        "I need to draft a counter-proposal for Pacific Rim Securities' CSA terms by Friday.",
        "Remember to send the updated netting opinion to Nordic Sovereign Fund's legal team by March 15.",
        "Action item: schedule a call with Apex Trading Group to discuss the NAV trigger levels.",
        "We must finalize the eligible collateral schedule for Sterling Asset Management before the quarter-end.",
        "Need to circulate the amended cross-default provisions to all BBB-rated counterparties for review.",
        "Deadline: submit the EMIR reporting reconciliation for all EU-domiciled counterparties by January 31.",
        "Going to prepare the initial margin calculation model for CFTC Phase 6 compliance by next month.",
        "Plan to review all force majeure provisions across the portfolio by end of Q1 2026.",
        "Must update the collateral eligibility matrix after the latest Basel III revisions take effect.",
        "Need to send the variation margin dispute resolution protocol to Sakura Financial Holdings.",
        "Action item: verify that all Singapore-domiciled counterparties have updated ISDA protocols.",
        "We need to complete the annual credit review for Condor Macro Fund before their rating watch expires.",
        "Todo: reconcile the threshold amounts across all CSAs against the current credit matrix.",
        "Must prepare the quarterly collateral adequacy report for the risk committee by March 20.",
    ]
    for i, text in enumerate(tasks):
        dt = date_in_month(2026, 1) + timedelta(days=i * 3)
        facts.append({"role": "user", "text": text, "date": dt})
    return facts


def generate_regulatory_facts() -> list[dict]:
    """Generate regulatory reference facts."""
    facts = []
    reg_facts = [
        "Under Basel III, the minimum leverage ratio for our counterparties is 3%, which affects the independent amount calculations for bank counterparties.",
        "EMIR Refit requires mandatory electronic reporting of all derivative transactions to a registered trade repository within 1 business day.",
        "The CFTC Phase 6 initial margin rules now apply to entities with an aggregate notional of $8 billion, which captures Meridian Capital Partners and Atlas Global Fund.",
        "Dodd-Frank Section 723 requires all standardized swaps to be cleared through a registered DCO, impacting our bilateral netting arrangements.",
        "The SEC's new margin rule under Rule 18a-3 requires broker-dealers to collect initial margin on non-cleared security-based swaps.",
        "Under EMIR, variation margin must be exchanged on a daily basis for all counterparties exceeding the EUR 8 billion threshold.",
        "Basel III SA-CCR methodology requires us to recalculate exposure-at-default for all derivative portfolios using the new add-on factors.",
        "The ISDA 2025 Protocol amends the definition of credit events to include governmental intervention as a new trigger category.",
        "CFTC Regulation 23.504 requires that all swap documentation be executed prior to or contemporaneously with the first trade.",
        "Under MiFID II, we need to ensure best execution reporting for all derivative transactions with EU-domiciled counterparties.",
        "The Basel III output floor phases in at 72.5% by 2028, which will increase capital requirements for our internal model-based counterparties.",
        "EMIR DORA requirements mandate operational resilience testing for all critical ICT systems supporting derivative clearing and settlement.",
        "Dodd-Frank Title VII requires us to maintain records of all swap transactions for a minimum of 5 years from the termination date.",
        "The ISDA Master Agreement Section 14 definitions are being updated to align with the new SOFR-based fallback provisions under the 2025 supplements.",
        "CFTC No-Action Letter 25-01 extends relief for certain reporting obligations for non-US counterparties until December 2026.",
    ]
    for i, text in enumerate(reg_facts):
        dt = random_date()
        facts.append({"role": "assistant", "text": text, "date": dt})
    return facts


def generate_market_data_facts() -> list[dict]:
    """Generate market data and valuation facts."""
    facts = []
    market_facts = [
        "Current USD 5-year swap rate is 3.85%, up 15 basis points from last week's fixing.",
        "The EUR/USD basis swap spread is -22 basis points for the 3-month tenor.",
        "Credit spreads for BBB-rated corporates have widened to 165 basis points over Treasuries.",
        "The 10-year US Treasury yield closed at 4.12% yesterday, its highest level since November.",
        "Implied volatility on 1-year EUR/USD at-the-money options is trading at 8.2%.",
        "The mark-to-market value of our portfolio with Meridian Capital Partners is -$12.4 million (we owe them).",
        "Net credit exposure to Pacific Rim Securities stands at $45.2 million after netting.",
        "The variation margin call to Apex Trading Group for today is $3.8 million based on the latest valuations.",
        "Total initial margin collected across all counterparties is $892 million, up $47 million from last month.",
        "The portfolio delta for our interest rate book is -$2.3 million per basis point.",
        "JPY/USD is trading at 148.50 with the Bank of Japan maintaining yield curve control.",
        "CDS spreads on Condor Macro Fund's reference entities have tightened 12 basis points to 285 basis points.",
        "The collateral portfolio haircut-adjusted value is $1.24 billion against total exposure of $1.18 billion.",
        "USD LIBOR 3-month fixing is at 5.42%, continuing its phase-out transition to SOFR at 5.33%.",
        "GBP 5-year gilt yields are at 4.05% following the Bank of England's latest rate decision.",
        "Our aggregate notional outstanding with all counterparties is $78.3 billion across interest rate and credit derivatives.",
        "The CVA adjustment for Atlas Global Fund increased by $1.2 million due to the rating outlook change.",
        "Collateral utilization ratio is at 94% across the entire CSA portfolio.",
        "The weighted average haircut on pledged collateral is 2.8%, dominated by US Treasuries.",
        "Mark-to-market on the Sterling Asset Management IRS portfolio is +$8.7 million in our favor.",
    ]
    for i, text in enumerate(market_facts):
        dt = random_date()
        facts.append({"role": "assistant", "text": text, "date": dt})
    return facts


def generate_contradiction_facts() -> list[dict]:
    """Generate facts that CONTRADICT earlier facts (threshold changes over time)."""
    facts = []

    # Meridian: threshold changed from $50M to $75M
    facts.append({"role": "assistant", "text": "The cross-default threshold for Meridian Capital Partners is $50 million under Section 5(a)(vi).", "date": date_in_month(2025, 9, )})
    facts.append({"role": "user", "text": "We increased the cross-default threshold for Meridian Capital Partners to $75 million after the latest credit review.", "date": date_in_month(2025, 12)})

    # Pacific Rim: rating downgrade
    facts.append({"role": "assistant", "text": "Pacific Rim Securities currently holds an A- credit rating from S&P.", "date": date_in_month(2025, 9)})
    facts.append({"role": "assistant", "text": "Pacific Rim Securities has been downgraded to BBB+ by S&P following the Asian credit review.", "date": date_in_month(2025, 11)})

    # Apex: MTA changed
    facts.append({"role": "assistant", "text": "The minimum transfer amount for Apex Trading Group is $1 million under the CSA.", "date": date_in_month(2025, 10)})
    facts.append({"role": "user", "text": "We agreed to reduce the minimum transfer amount for Apex Trading Group from $1 million to $500,000.", "date": date_in_month(2026, 1)})

    # Atlas: collateral changed
    facts.append({"role": "assistant", "text": "Eligible collateral for Atlas Global Fund is limited to cash in USD and EUR only.", "date": date_in_month(2025, 9)})
    facts.append({"role": "user", "text": "We expanded eligible collateral for Atlas Global Fund to include US Treasuries with a 2% haircut alongside cash.", "date": date_in_month(2026, 2)})

    # Condor: NAV trigger changed
    facts.append({"role": "assistant", "text": "The NAV trigger for Condor Macro Fund is set at 30% decline as an Additional Termination Event.", "date": date_in_month(2025, 10)})
    facts.append({"role": "user", "text": "We tightened the NAV trigger for Condor Macro Fund from 30% to 20% decline following their recent volatility.", "date": date_in_month(2026, 1)})

    return facts


def generate_near_duplicate_facts() -> list[dict]:
    """Generate near-duplicate facts (same info, different phrasing)."""
    facts = []
    dt = random_date()

    # Pair 1: same threshold, different wording
    facts.append({"role": "assistant", "text": "Nordic Sovereign Fund has a zero threshold under the CSA given their sovereign status.", "date": dt})
    facts.append({"role": "user", "text": "The threshold for Nordic Sovereign Fund is set to zero in the Credit Support Annex because they're a sovereign entity.", "date": dt + timedelta(days=1)})

    # Pair 2: same collateral info
    facts.append({"role": "assistant", "text": "Sterling Asset Management's CSA eligible collateral includes cash in USD/EUR/GBP and US Treasuries at 2% haircut.", "date": dt + timedelta(days=2)})
    facts.append({"role": "user", "text": "For Sterling Asset Management, eligible collateral under the Credit Support Annex is cash in major currencies plus Treasuries with a 2% haircut.", "date": dt + timedelta(days=3)})

    # Pair 3: same netting provision
    facts.append({"role": "assistant", "text": "Rhine Valley Bank uses full close-out netting under Section 6(e) of the ISDA Master Agreement.", "date": dt + timedelta(days=4)})
    facts.append({"role": "assistant", "text": "Close-out netting applies to Rhine Valley Bank under ISDA Section 6(e) with bilateral payment netting.", "date": dt + timedelta(days=5)})

    # Pair 4: same rating info
    facts.append({"role": "assistant", "text": "Sakura Financial Holdings maintains an A+ credit rating from both S&P and Moody's.", "date": dt + timedelta(days=6)})
    facts.append({"role": "user", "text": "The credit rating for Sakura Financial Holdings is A+ according to S&P and Moody's latest assessment.", "date": dt + timedelta(days=7)})

    # Pair 5: same regulatory fact
    facts.append({"role": "assistant", "text": "CFTC Phase 6 initial margin requirements apply to entities with aggregate notional exceeding $8 billion.", "date": dt + timedelta(days=8)})
    facts.append({"role": "assistant", "text": "Under CFTC Phase 6, initial margin must be posted by counterparties whose aggregate average notional amount exceeds $8 billion.", "date": dt + timedelta(days=9)})

    return facts


def generate_noise_facts() -> list[dict]:
    """Generate conversational noise that should be filtered."""
    facts = []
    noise = [
        "Sure, sounds good.",
        "Ok thanks for that.",
        "Yes, let's proceed.",
        "Hello, can you help me with something?",
        "Got it, makes sense.",
        "Right, exactly.",
        "Thanks for the update.",
        "Understood, I'll take a look.",
        "Perfect, that's what I expected.",
        "Great, let's move on.",
        "Hmm, let me think about that.",
        "Fair enough.",
        "Interesting point.",
        "Can you help me with this?",
        "What do you think about that?",
        "Yes absolutely.",
        "No that's fine.",
        "Yep, confirmed.",
        "Ok sounds good, thanks.",
        "Hey, quick question.",
        "Let me help you with that. What would you like to know?",
        "I can help with that review.",
        "Sure, I'll take a look at that for you.",
        "That's a great question. Let me check.",
        "I'd be happy to assist with that.",
    ]
    for text in noise:
        facts.append({"role": random.choice(["user", "assistant"]), "text": text, "date": random_date()})
    return facts


def generate_additional_bulk_facts() -> list[dict]:
    """Generate additional facts to push past 500 total."""
    facts = []

    # More ISDA provision details per counterparty
    provisions = [
        "The governing law for the {cp} ISDA Master Agreement is English law with jurisdiction in London courts.",
        "Process agent for {cp} in the ISDA is designated as CT Corporation in New York.",
        "The calculation agent under the {cp} ISDA is specified as the Determining Party as defined in Section 14.",
        "Section 4(a) tax representations for {cp} include a complete Payee Tax Representation with W-8BEN-E.",
        "The {cp} agreement includes a waiver of jury trial under the Multibranch provisions.",
        "Transfer provisions for {cp} require prior written consent and are restricted to affiliates with equivalent credit rating.",
        "The cure period for an Event of Default under the {cp} ISDA is 3 business days for payment failures.",
        "Interest rate on overdue amounts for {cp} is the Federal Funds rate plus 1% per annum.",
    ]
    for cp_info in COUNTERPARTIES:
        cp = cp_info["name"]
        for template in random.sample(provisions, k=random.randint(4, 6)):
            dt = random_date()
            facts.append({"role": "assistant", "text": template.format(cp=cp), "date": dt})

    # Portfolio-level observations
    portfolio_facts = [
        "Our total derivative exposure across all 20 counterparties is $78.3 billion notional outstanding.",
        "The weighted average credit rating of the counterparty portfolio is BBB+ based on notional-weighted ratings.",
        "Concentration risk: top 3 counterparties represent 45% of total exposure by notional amount.",
        "Geographic distribution: 40% UK/Europe, 30% Americas, 25% Asia-Pacific, 5% Middle East.",
        "Product mix across the portfolio: 60% interest rate swaps, 25% credit derivatives, 10% FX forwards, 5% equity options.",
        "Average tenor of outstanding trades is 4.2 years with a range from 3 months to 30 years.",
        "Total collateral held under all CSAs is $1.24 billion, with 78% in cash and 22% in securities.",
        "The aggregate CVA across the portfolio is $34.5 million, up $3.2 million from last quarter.",
        "Wrong-way risk exposure is concentrated in 3 counterparties where credit and market risk are correlated.",
        "The portfolio has 847 open transactions across all counterparties, down from 912 at the start of the quarter.",
    ]
    for text in portfolio_facts:
        facts.append({"role": "assistant", "text": text, "date": random_date()})

    # More detailed regulatory compliance items
    compliance_facts = [
        "All EU-domiciled counterparties must report under EMIR to an approved trade repository within T+1.",
        "The ISDA DF Protocol adherence status: 18 of 20 counterparties have adhered, pending Shanghai Dragon Capital and Caspian Energy Fund.",
        "Basel III CVA capital charge applies to all non-centrally-cleared derivative exposures above the EUR 100 million threshold.",
        "Under SFTR, securities financing transactions with Rhine Valley Bank must be reported to an EU trade repository.",
        "MiFID II best execution obligations apply to all derivative transactions executed with EU-regulated counterparties.",
        "The LEI (Legal Entity Identifier) renewal is due for 4 counterparties in Q1 2026.",
        "FATCA reporting obligations apply to all US-source payments made to non-US counterparties in the portfolio.",
        "The CRS (Common Reporting Standard) requires annual exchange of financial account information for all non-resident counterparties.",
        "ISDA's IBOR Fallback Protocol has been adhered to by all 20 counterparties for LIBOR transition.",
        "The CFTC's real-time public reporting requirement under Part 43 applies to all USD-denominated interest rate swaps above $250 million notional.",
    ]
    for text in compliance_facts:
        facts.append({"role": "assistant", "text": text, "date": random_date()})

    # Additional counterparty-specific negotiation history
    negotiation_templates = [
        "{cp} requested a bilateral early termination clause, which we rejected pending legal review.",
        "The legal opinion for netting enforceability in {jurisdiction} was obtained for {cp} from Clifford Chance.",
        "{cp}'s credit officer confirmed their internal credit limit for us is {limit}.",
        "We received the executed ISDA schedule from {cp} on {date_str}, pending CSA finalization.",
        "The operational contact at {cp} for margin calls is their treasury desk in {jurisdiction}.",
        "{cp} uses TriOptima for portfolio reconciliation on a monthly cycle.",
        "Dispute resolution under the {cp} ISDA follows the ISDA Determination Committee process.",
        "The {cp} ISDA includes a set-off provision under Section 6(f) applicable to all affiliates.",
        "{cp} has elected for physical settlement on all credit derivative transactions.",
        "The master confirmation agreement with {cp} covers interest rate swaps denominated in USD, EUR, and GBP.",
    ]
    limits = ["$200 million", "$500 million", "$1 billion", "$150 million", "$300 million"]
    for cp_info in COUNTERPARTIES:
        cp = cp_info["name"]
        jurisdiction = cp_info["jurisdiction"]
        for template in random.sample(negotiation_templates, k=random.randint(3, 5)):
            dt = random_date()
            text = template.format(
                cp=cp,
                jurisdiction=jurisdiction,
                limit=random.choice(limits),
                date_str=dt.strftime("%B %d, %Y"),
            )
            facts.append({"role": random.choice(["user", "assistant"]), "text": text, "date": dt})

    return facts


def generate_all_facts() -> list[dict]:
    """Generate all test facts and return sorted by date."""
    all_facts = []
    all_facts.extend(generate_counterparty_facts())
    all_facts.extend(generate_isda_facts())
    all_facts.extend(generate_csa_facts())
    all_facts.extend(generate_decision_facts())
    all_facts.extend(generate_task_facts())
    all_facts.extend(generate_regulatory_facts())
    all_facts.extend(generate_market_data_facts())
    all_facts.extend(generate_contradiction_facts())
    all_facts.extend(generate_near_duplicate_facts())
    all_facts.extend(generate_noise_facts())
    all_facts.extend(generate_additional_bulk_facts())

    # Sort by date to simulate chronological ingestion
    all_facts.sort(key=lambda f: f["date"])
    return all_facts


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Ingest
# ══════════════════════════════════════════════════════════════════

def run_ingestion(facts: list[dict]) -> dict:
    """Ingest all facts into a fresh daemon, return timing stats."""
    # Clean previous test data
    for path in [VAULT_PATH, DB_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(VAULT_PATH, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)

    daemon = MemoryDaemon(vault_path=VAULT_PATH, db_path=DB_PATH, session_id="scale-test")

    # We bypass the async queue and use the extractor + store directly
    # for precise timing. The daemon.ingest() is async and would require
    # waiting for the queue to drain with no deterministic signal.
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[1]

    t_start = time.time()
    batch_times = []
    total_extracted = 0
    total_stored = 0
    total_deduped = 0
    batch_size = 50

    for i in range(0, len(facts), batch_size):
        batch = facts[i:i + batch_size]
        t_batch = time.time()

        for item in batch:
            # Temporarily override the timestamp in the extractor output
            extracted = daemon.extractor.extract(item["text"], role=item["role"])
            for fact in extracted:
                fact["session"] = "scale-test"
                # Override timestamp to match our simulated date
                fact["timestamp"] = item["date"].isoformat()

                fact_id = daemon.store.store(fact)
                if fact_id:
                    total_stored += 1
                    daemon.vault.write_fact(fact)
                else:
                    total_deduped += 1
            total_extracted += len(extracted)

        batch_time = time.time() - t_batch
        batch_times.append(batch_time)

    t_total = time.time() - t_start
    mem_after = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Keep daemon reference for recall phase
    return {
        "daemon": daemon,
        "total_time": t_total,
        "batch_times": batch_times,
        "total_facts_input": len(facts),
        "total_extracted": total_extracted,
        "total_stored": total_stored,
        "total_deduped": total_deduped,
        "facts_per_second": total_extracted / t_total if t_total > 0 else 0,
        "mem_delta_mb": (mem_after - mem_before) / (1024 * 1024),
        "db_count": daemon.store.count(),
    }


# ══════════════════════════════════════════════════════════════════
# PHASE 3: Retrieval quality benchmark
# ══════════════════════════════════════════════════════════════════

def build_test_queries() -> list[dict]:
    """Build 20 test queries with expected results and validation criteria."""
    return [
        # --- Contradiction tests (should return LATEST value) ---
        {
            "query": "What is the cross-default threshold for Meridian Capital Partners?",
            "expected_substring": "$75 million",
            "contradicted_substring": "$50 million",
            "category": "contradiction",
            "description": "Should return updated $75M threshold, not old $50M",
        },
        {
            "query": "What is Pacific Rim Securities credit rating?",
            "expected_substring": "BBB+",
            "contradicted_substring": "A-",
            "category": "contradiction",
            "description": "Should return downgraded BBB+, not old A-",
        },
        {
            "query": "What is the minimum transfer amount for Apex Trading Group?",
            "expected_substring": "$500,000",
            "contradicted_substring": "$1 million",
            "category": "contradiction",
            "description": "Should return reduced $500K MTA, not old $1M",
        },
        {
            "query": "What collateral is eligible for Atlas Global Fund?",
            "expected_substring": "US Treasuries",
            "contradicted_substring": "cash in USD and EUR only",
            "category": "contradiction",
            "description": "Should return expanded collateral including Treasuries",
        },
        {
            "query": "What is the NAV trigger for Condor Macro Fund?",
            "expected_substring": "20%",
            "contradicted_substring": "30%",
            "category": "contradiction",
            "description": "Should return tightened 20% trigger, not old 30%",
        },
        # --- Specific entity retrieval ---
        {
            "query": "Tell me about Nordic Sovereign Fund's CSA threshold",
            "expected_substring": "zero",
            "contradicted_substring": None,
            "category": "entity_lookup",
            "description": "Should find zero threshold for AAA sovereign",
        },
        {
            "query": "What is Sakura Financial Holdings credit rating?",
            "expected_substring": "A+",
            "contradicted_substring": None,
            "category": "entity_lookup",
            "description": "Should find A+ rating",
        },
        {
            "query": "What sector is Summit Peak Holdings in?",
            "expected_substring": "Family Office",
            "contradicted_substring": None,
            "category": "entity_lookup",
            "description": "Should find Family Office sector",
        },
        # --- Type-filtered queries ---
        {
            "query": "What decisions were made about cross-default thresholds?",
            "expected_substring": "decided",
            "contradicted_substring": None,
            "category": "type_filter",
            "description": "Should find decision-type facts about thresholds",
        },
        {
            "query": "What tasks are pending for ISDA reviews?",
            "expected_substring": "need to",
            "contradicted_substring": None,
            "category": "type_filter",
            "description": "Should find task-type facts about reviews",
        },
        # --- Multi-result queries ---
        {
            "query": "Which counterparties are BBB rated?",
            "expected_substring": "BBB",
            "contradicted_substring": None,
            "category": "multi_result",
            "description": "Should return multiple BBB-rated counterparties",
        },
        {
            "query": "What are the eligible collateral types across counterparties?",
            "expected_substring": "collateral",
            "contradicted_substring": None,
            "category": "multi_result",
            "description": "Should return multiple collateral specifications",
        },
        # --- Regulatory queries ---
        {
            "query": "What are the CFTC initial margin requirements?",
            "expected_substring": "CFTC",
            "contradicted_substring": None,
            "category": "regulatory",
            "description": "Should find CFTC Phase 6 margin rules",
        },
        {
            "query": "What are the EMIR reporting obligations?",
            "expected_substring": "EMIR",
            "contradicted_substring": None,
            "category": "regulatory",
            "description": "Should find EMIR reporting requirements",
        },
        {
            "query": "What Basel III requirements affect our portfolio?",
            "expected_substring": "Basel",
            "contradicted_substring": None,
            "category": "regulatory",
            "description": "Should find Basel III capital/leverage requirements",
        },
        # --- Market data queries ---
        {
            "query": "What is the current swap rate?",
            "expected_substring": "swap rate",
            "contradicted_substring": None,
            "category": "market_data",
            "description": "Should find USD swap rate data",
        },
        {
            "query": "What is our exposure to Meridian Capital Partners?",
            "expected_substring": "Meridian",
            "contradicted_substring": None,
            "category": "market_data",
            "description": "Should find MTM or exposure data for Meridian",
        },
        # --- Cross-cutting queries ---
        {
            "query": "What netting provisions apply to our counterparties?",
            "expected_substring": "netting",
            "contradicted_substring": None,
            "category": "cross_cutting",
            "description": "Should find netting-related facts across counterparties",
        },
        {
            "query": "What are the force majeure provisions?",
            "expected_substring": "Force Majeure",
            "contradicted_substring": None,
            "category": "cross_cutting",
            "description": "Should find force majeure terms",
        },
        {
            "query": "What is the total portfolio notional outstanding?",
            "expected_substring": "$78.3 billion",
            "contradicted_substring": None,
            "category": "cross_cutting",
            "description": "Should find aggregate portfolio notional",
        },
    ]


def run_retrieval_benchmark(daemon: MemoryDaemon) -> list[dict]:
    """Run all 20 test queries and measure quality metrics."""
    queries = build_test_queries()
    results = []

    for q in queries:
        t_start = time.time()
        recalled = daemon.store.recall(q["query"], n_results=5, recency_weight=0.15)
        latency_ms = (time.time() - t_start) * 1000

        top3_texts = [r["text"] for r in recalled[:3]]
        top5_texts = [r["text"] for r in recalled[:5]]
        top3_joined = " ".join(top3_texts).lower()
        top5_joined = " ".join(top5_texts).lower()

        # Check if expected substring is in top 3
        expected_in_top3 = q["expected_substring"].lower() in top3_joined
        expected_in_top5 = q["expected_substring"].lower() in top5_joined

        # Check for contradiction leakage
        contradiction_leaked = False
        contradiction_rank = None
        if q["contradicted_substring"]:
            for i, text in enumerate(top5_texts):
                if q["contradicted_substring"].lower() in text.lower():
                    # Check if the contradicted value appears WITHOUT the updated value
                    # i.e., it's the OLD fact, not the fact that describes the change
                    if q["expected_substring"].lower() not in text.lower():
                        contradiction_leaked = True
                        contradiction_rank = i + 1
                        break

        # Check if expected result ranks above contradicted result
        expected_rank = None
        contradicted_rank_val = None
        if q["contradicted_substring"]:
            for i, text in enumerate(top5_texts):
                text_lower = text.lower()
                if expected_rank is None and q["expected_substring"].lower() in text_lower:
                    expected_rank = i + 1
                if contradicted_rank_val is None and q["contradicted_substring"].lower() in text_lower:
                    if q["expected_substring"].lower() not in text_lower:
                        contradicted_rank_val = i + 1

        results.append({
            "query": q["query"],
            "description": q["description"],
            "category": q["category"],
            "expected_in_top3": expected_in_top3,
            "expected_in_top5": expected_in_top5,
            "contradiction_leaked": contradiction_leaked,
            "contradiction_rank": contradiction_rank,
            "expected_rank": expected_rank,
            "contradicted_rank": contradicted_rank_val,
            "latency_ms": latency_ms,
            "top3_texts": top3_texts,
            "top_score": recalled[0]["score"] if recalled else 0,
        })

    return results


# ══════════════════════════════════════════════════════════════════
# PHASE 4: Summary and reporting
# ══════════════════════════════════════════════════════════════════

def print_results(ingest_stats: dict, retrieval_results: list[dict], dedup_count: int):
    """Print comprehensive results report."""
    W = 80

    print()
    print("=" * W)
    print("  MEMORY DAEMON SCALE TEST RESULTS")
    print("=" * W)

    # ── Ingestion stats ──
    print()
    print("-" * W)
    print("  PHASE 2: INGESTION")
    print("-" * W)
    print(f"  Total facts generated:    {ingest_stats['total_facts_input']}")
    print(f"  Facts extracted:          {ingest_stats['total_extracted']}")
    print(f"  Facts stored (unique):    {ingest_stats['total_stored']}")
    print(f"  Facts deduped:            {ingest_stats['total_deduped']}")
    print(f"  DB collection count:      {ingest_stats['db_count']}")
    print(f"  Total ingestion time:     {ingest_stats['total_time']:.2f}s")
    print(f"  Throughput:               {ingest_stats['facts_per_second']:.1f} facts/sec")
    print(f"  Memory delta:             {ingest_stats['mem_delta_mb']:.1f} MB")
    if ingest_stats['batch_times']:
        avg_batch = sum(ingest_stats['batch_times']) / len(ingest_stats['batch_times'])
        print(f"  Avg batch time (50):      {avg_batch:.3f}s")
        print(f"  Slowest batch:            {max(ingest_stats['batch_times']):.3f}s")
        print(f"  Fastest batch:            {min(ingest_stats['batch_times']):.3f}s")

    # ── Retrieval results ──
    print()
    print("-" * W)
    print("  PHASE 3: RETRIEVAL QUALITY")
    print("-" * W)

    # Per-query results table
    print()
    print(f"  {'#':<3} {'Category':<15} {'Top3':<5} {'Top5':<5} {'Leak':<5} {'ms':<8} Description")
    print(f"  {'─'*3} {'─'*15} {'─'*5} {'─'*5} {'─'*5} {'─'*8} {'─'*35}")

    for i, r in enumerate(retrieval_results):
        top3 = "Y" if r["expected_in_top3"] else "N"
        top5 = "Y" if r["expected_in_top5"] else "N"
        leak = "Y" if r["contradiction_leaked"] else ("n/a" if r["category"] != "contradiction" else "N")
        print(f"  {i+1:<3} {r['category']:<15} {top3:<5} {top5:<5} {leak:<5} {r['latency_ms']:<8.1f} {r['description'][:35]}")

    # ── Aggregate metrics ──
    print()
    print("-" * W)
    print("  AGGREGATE METRICS")
    print("-" * W)

    total_q = len(retrieval_results)
    precision_3 = sum(1 for r in retrieval_results if r["expected_in_top3"]) / total_q
    precision_5 = sum(1 for r in retrieval_results if r["expected_in_top5"]) / total_q

    contradiction_queries = [r for r in retrieval_results if r["category"] == "contradiction"]
    contradiction_count = len(contradiction_queries)
    leaks = sum(1 for r in contradiction_queries if r["contradiction_leaked"])
    leak_rate = leaks / contradiction_count if contradiction_count > 0 else 0

    # Contradiction ordering: how often does the LATEST fact rank above the old one?
    correct_order = 0
    order_testable = 0
    for r in contradiction_queries:
        if r["expected_rank"] is not None and r["contradicted_rank"] is not None:
            order_testable += 1
            if r["expected_rank"] < r["contradicted_rank"]:
                correct_order += 1
        elif r["expected_rank"] is not None and r["contradicted_rank"] is None:
            # Old fact not in top 5 at all — good
            order_testable += 1
            correct_order += 1

    order_rate = correct_order / order_testable if order_testable > 0 else 0

    avg_latency = sum(r["latency_ms"] for r in retrieval_results) / total_q
    max_latency = max(r["latency_ms"] for r in retrieval_results)
    min_latency = min(r["latency_ms"] for r in retrieval_results)

    dedup_rate = ingest_stats["total_deduped"] / (ingest_stats["total_stored"] + ingest_stats["total_deduped"]) if (ingest_stats["total_stored"] + ingest_stats["total_deduped"]) > 0 else 0

    print(f"  Precision@3:              {precision_3:.1%} ({sum(1 for r in retrieval_results if r['expected_in_top3'])}/{total_q})")
    print(f"  Precision@5:              {precision_5:.1%} ({sum(1 for r in retrieval_results if r['expected_in_top5'])}/{total_q})")
    print(f"  Contradiction leakage:    {leak_rate:.1%} ({leaks}/{contradiction_count} queries leaked old values)")
    print(f"  Recency ordering:         {order_rate:.1%} ({correct_order}/{order_testable} latest fact ranked higher)")
    print(f"  Dedup rate:               {dedup_rate:.1%} ({ingest_stats['total_deduped']} deduped of {ingest_stats['total_stored'] + ingest_stats['total_deduped']} total)")
    print(f"  Avg retrieval latency:    {avg_latency:.1f} ms")
    print(f"  Max retrieval latency:    {max_latency:.1f} ms")
    print(f"  Min retrieval latency:    {min_latency:.1f} ms")

    # ── Overall score ──
    print()
    print("-" * W)
    print("  OVERALL SCORE")
    print("-" * W)

    # Weighted scoring
    scores = {
        "Precision@3": (precision_3, 0.30),
        "Precision@5": (precision_5, 0.20),
        "No contradiction leakage": (1 - leak_rate, 0.20),
        "Recency ordering": (order_rate, 0.15),
        "Latency (<50ms avg)": (min(1.0, 50 / avg_latency) if avg_latency > 0 else 1.0, 0.10),
        "Dedup effectiveness": (min(1.0, dedup_rate / 0.05) if dedup_rate > 0 else 0, 0.05),
    }

    overall = 0
    for name, (score, weight) in scores.items():
        weighted = score * weight
        overall += weighted
        bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
        print(f"  {name:<28} [{bar}] {score:.1%} (x{weight:.2f} = {weighted:.3f})")

    print()
    print(f"  OVERALL SCORE: {overall:.1%}")

    if overall >= 0.8:
        grade = "EXCELLENT"
    elif overall >= 0.6:
        grade = "GOOD"
    elif overall >= 0.4:
        grade = "FAIR"
    else:
        grade = "NEEDS WORK"
    print(f"  GRADE: {grade}")

    # ── Degradation patterns ──
    print()
    print("-" * W)
    print("  DEGRADATION ANALYSIS")
    print("-" * W)

    # Check if contradiction queries perform worse
    contradiction_p3 = sum(1 for r in contradiction_queries if r["expected_in_top3"]) / len(contradiction_queries) if contradiction_queries else 0
    other_queries = [r for r in retrieval_results if r["category"] != "contradiction"]
    other_p3 = sum(1 for r in other_queries if r["expected_in_top3"]) / len(other_queries) if other_queries else 0

    print(f"  Contradiction queries P@3:  {contradiction_p3:.1%}")
    print(f"  Other queries P@3:          {other_p3:.1%}")
    if contradiction_p3 < other_p3 - 0.1:
        print("  WARNING: Contradiction queries significantly underperform. Recency weighting may need tuning.")
    else:
        print("  OK: No significant degradation on contradiction queries.")

    # Check latency distribution
    by_category = {}
    for r in retrieval_results:
        by_category.setdefault(r["category"], []).append(r["latency_ms"])
    print()
    print("  Latency by category:")
    for cat, lats in sorted(by_category.items()):
        avg = sum(lats) / len(lats)
        print(f"    {cat:<20} avg={avg:.1f}ms")

    print()
    print("=" * W)
    print()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    random.seed(42)  # Reproducible results

    print()
    print("+" + "-" * 68 + "+")
    print("|  MEMORY DAEMON SCALE TEST                                        |")
    print("|  500+ ISDA/finance facts, 20 counterparties, 6 months            |")
    print("+" + "-" * 68 + "+")

    # ── Phase 1: Generate ──
    print()
    print("[Phase 1] Generating realistic ISDA/finance facts...")
    facts = generate_all_facts()
    print(f"  Generated {len(facts)} fact entries across 6 months")
    print(f"  Date range: {facts[0]['date'].strftime('%Y-%m-%d')} to {facts[-1]['date'].strftime('%Y-%m-%d')}")

    # Count by type
    noise_count = sum(1 for f in facts if len(f["text"]) < 30)
    print(f"  Includes: ~{noise_count} noise entries, 10 contradiction pairs, 10 near-duplicate pairs")

    # ── Phase 2: Ingest ──
    print()
    print("[Phase 2] Ingesting into memory daemon...")
    print(f"  Vault: {VAULT_PATH}")
    print(f"  DB:    {DB_PATH}")
    ingest_stats = run_ingestion(facts)
    print(f"  Done. {ingest_stats['total_stored']} facts stored in {ingest_stats['total_time']:.2f}s")

    # ── Phase 3: Retrieval benchmark ──
    print()
    print("[Phase 3] Running retrieval quality benchmark (20 queries)...")
    daemon = ingest_stats["daemon"]
    retrieval_results = run_retrieval_benchmark(daemon)
    print(f"  Done. All 20 queries executed.")

    # ── Phase 4: Report ──
    print_results(ingest_stats, retrieval_results, ingest_stats["total_deduped"])


if __name__ == "__main__":
    main()
