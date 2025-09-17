# copart_bid_calculator.py
"""
Copart Auction Max-Bid Calculator (Single-file Streamlit app)

How to run:
 - (Optional) export OPENAI_API_KEY="sk-..."
 - In terminal: pip install streamlit
 - Run: streamlit run copart_bid_calculator.py

This file:
 - auto-installs missing packages when run in a notebook/VSCode environment
 - tries to fetch Copart lot details from a Copart URL (best-effort)
 - integrates exchangerate.host for FX rates
 - optionally calls OpenAI (if OPENAI_API_KEY env var is set) to estimate repair/resale costs
 - presents a clean Streamlit UI and sensitivity analysis
"""

# --------- Auto-install required packages if missing (helps Jupyter/VSCode plug & play) ----------
import importlib, subprocess, sys, os, json, math
required = ["streamlit", "requests", "bs4", "pandas", "openai"]
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# --------- Imports ----------
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import openai
from typing import Optional, Dict, Any

# --------- Constants & Helpers ----------
EXCHANGE_API = "https://api.exchangerate.host/convert"
DEFAULT_AUCTION_CURRENCY = "USD"
st.set_page_config(page_title="Copart Max Bid Calculator", layout="wide")

def safe_json_request(url, params=None, headers=None, timeout=12):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def get_fx_rate(from_ccy: str, to_ccy: str) -> Optional[float]:
    """Get FX via exchangerate.host (free). Returns rate (1 from_ccy = rate to_ccy)."""
    if from_ccy.upper() == to_ccy.upper():
        return 1.0
    try:
        params = {"from": from_ccy.upper(), "to": to_ccy.upper()}
        data = safe_json_request(EXCHANGE_API, params=params)
        if data and "result" in data:
            return float(data["result"])
    except Exception:
        pass
    return None

def parse_copart_page(url: str) -> Dict[str, Any]:
    """
    Attempt to parse simple fields from a Copart lot page.
    NOTE: Copart pages are often dynamic; this is best-effort. Provide manual override UI.
    """
    out = {}
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible)"}  # polite
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Best-effort selectors; may change. We'll try to extract title, year, make, model, damage, odometer.
        title = soup.find("h1")
        if title:
            out["title"] = title.get_text(strip=True)
        # Look for keywords
        text = soup.get_text(" ", strip=True)
        for key in ["Year", "Make", "Model", "Odometer", "VIN", "Damage"]:
            # cheap heuristic: find "Year: 2018" patterns
            idx = text.find(key)
            if idx != -1:
                snippet = text[idx:idx+80]
                out[key.lower()] = snippet
        # may attempt to find current bid
        bid = None
        # Copart often hides financials; skip aggressive scraping
        out["fetched"] = True
    except Exception as e:
        out["fetched"] = False
        out["error"] = str(e)
    return out

def ask_openai_for_estimates(openai_api_key: str, prompt: str) -> Optional[str]:
    """Call OpenAI (ChatGPT) to get text response. Expects OPENAI_API_KEY in env or passed key."""
    if not openai_api_key:
        return None
    try:
        openai.api_key = openai_api_key
        # Use a short ChatCompletion (gpt-4 style or gpt-3.5-turbo)
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list() else "gpt-4",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=800
        )
        text = resp["choices"][0]["message"]["content"]
        return text
    except Exception as e:
        # fallback: try gpt-3.5-turbo
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=800
            )
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return None

# --------- Core calculation ----------
def compute_max_bid(resale_price_dest_ccy: float,
                    resale_ccy_to_auction_rate: float,
                    repair_cost_auction_ccy: float,
                    logistics_cost_auction_ccy: float,
                    other_fees_auction_ccy: float,
                    desired_profit_auction_ccy: float) -> float:
    """
    Compute recommended max bid (in auction currency).
    Formula:
      max_bid = resale_price_in_auction_ccy - (repair + logistics + other_fees + desired_profit)
    """
    resale_in_auction = resale_price_dest_ccy * resale_ccy_to_auction_rate
    total_costs = repair_cost_auction_ccy + logistics_cost_auction_ccy + other_fees_auction_ccy + desired_profit_auction_ccy
    max_bid = resale_in_auction - total_costs
    return max(0.0, max_bid)

def percent_change_table(base, variations=[-0.1, 0.1]):
    rows = []
    for v in variations:
        rows.append({"scenario": f"{int(v*100)}%", "value": base * (1+v)})
    return pd.DataFrame(rows)

# --------- Streamlit UI ----------
st.title("Copart Auction — Max Bid / Target Profit Calculator")
st.caption("Estimate the maximum bid to place on a Copart lot so you achieve a target profit after FX, repairs, logistics, fees and taxes.")

# Sidebar: settings
st.sidebar.header("Settings & Integrations")
openai_key = st.sidebar.text_input("OPENAI API Key (optional)", value=os.getenv("OPENAI_API_KEY") or "", type="password")
use_openai = st.sidebar.checkbox("Enable OpenAI for repair/resale estimates (costs may apply)", value=bool(openai_key))
auction_currency = st.sidebar.selectbox("Auction currency (Copart lot currency)", ["USD","CAD","GBP","EUR"], index=0)

# Main inputs
with st.form("inputs"):
    st.subheader("Lot & Route")
    copart_url = st.text_input("Copart Lot URL or Lot number (optional)")
    origin_country = st.selectbox("Origin country (where car is bought)", ["United States","United Kingdom","Japan","Germany","Canada"], index=0)
    destination_country = st.selectbox("Destination country (where car will be repaired/sold)", ["Nigeria","Ghana","USA","UK"], index=0)

    st.markdown("### Car details (auto-fill if fetch succeeded; you can edit)")
    fetched = {}
    car_title = st.text_input("Title / Trim / Notes", value="")
    year = st.text_input("Year", value="")
    make = st.text_input("Make", value="")
    model = st.text_input("Model", value="")
    mileage = st.text_input("Odometer (units as written)", value="")
    condition_notes = st.text_area("Condition / Damage notes", value="")
    est_current_bid = st.number_input("Current visible price / estimated bid (in auction currency)", min_value=0.0, value=0.0, step=100.0)

    st.markdown("### Expected financial targets")
    profit_mode = st.selectbox("Desired profit (absolute or percent)", ["Absolute (destination currency)","Absolute (auction currency)","Percent of resale price (%)"], index=0)
    profit_value = st.number_input("Desired profit value (enter number as per selected mode)", min_value=0.0, value=1000.0, step=100.0)

    # Costs inputs with heuristics
    st.markdown("### Manual cost overrides & heuristics")
    default_repair = st.number_input("Estimated repair cost (in auction currency) [override]", min_value=0.0, value=2000.0, step=100.0)
    trucking_to_port = st.number_input("Trucking from Copart to export port (auction currency)", min_value=0.0, value=300.0, step=50.0)
    copart_fees = st.number_input("Copart fees (buyer's fees) (auction currency)", min_value=0.0, value=400.0, step=50.0)
    shipping_cost = st.number_input("Ocean shipping to destination port (auction currency)", min_value=0.0, value=1500.0, step=50.0)
    import_clearance = st.number_input("Import duties & clearance (destination currency)", min_value=0.0, value=2000.0, step=100.0)
    local_trucking = st.number_input("Local trucking / port fees in destination (destination currency)", min_value=0.0, value=200.0, step=50.0)
    local_sale_fees = st.number_input("Local sales commission / taxes (destination currency)", min_value=0.0, value=300.0, step=50.0)

    st.markdown("### Resale / Valuation")
    resale_price_dest_ccy = st.number_input(f"Expected resale price in destination currency ({destination_country} local)", min_value=0.0, value=8000.0, step=100.0)

    submitted = st.form_submit_button("Calculate max bid")

# Attempt to fetch Copart info (best-effort) if URL provided
if copart_url and submitted:
    parsed = parse_copart_page(copart_url)
    if parsed.get("fetched"):
        st.success("Attempted to fetch Copart page content (best-effort). Please verify fields below.")
        # Tentatively populate fields (we won't overwrite what user typed)
        if parsed.get("title") and not car_title:
            car_title = parsed.get("title")
        # Show fetched keys
        st.json(parsed)
    else:
        st.warning("Could not reliably fetch Copart data; please enter details manually. Error: " + parsed.get("error", "unknown"))

# Main computation (only after submit)
if submitted:
    st.header("Results & Analysis")

    # Resolve desired profit to auction currency
    # First, resolve FX between destination currency and auction currency
    # We approximate destination currency code by some heuristics:
    dest_ccy_map = {"Nigeria":"NGN", "Ghana":"GHS", "USA":"USD", "UK":"GBP"}
    dest_ccy = dest_ccy_map.get(destination_country, "USD")
    auction_ccy = auction_currency

    fx_rate_dest_to_auction = get_fx_rate(dest_ccy, auction_ccy) or 1.0  # 1 dest = X auction
    fx_rate_auction_to_dest = get_fx_rate(auction_ccy, dest_ccy) or (1.0/fx_rate_dest_to_auction if fx_rate_dest_to_auction else 1.0)

    st.write(f"FX: 1 {dest_ccy} = {fx_rate_dest_to_auction:.4f} {auction_ccy}")

    # If user provided profit in percent or destination currency, convert to auction ccy
    if profit_mode == "Absolute (destination currency)":
        desired_profit_auction = profit_value * fx_rate_dest_to_auction
    elif profit_mode == "Absolute (auction currency)":
        desired_profit_auction = profit_value
    else:  # percent
        desired_profit_auction = (profit_value/100.0) * resale_price_dest_ccy * fx_rate_dest_to_auction

    # Repair cost: use OpenAI if enabled AND user left the default repair input untouched or chose to use AI
    repair_cost_auction = default_repair
    openai_notes = None
    if use_openai and openai_key:
        st.info("Requesting OpenAI to produce repair and resale adjustments. This will use your API key and may incur cost.")
        prompt = f"""
Estimate repair cost and expected resale price for the following car sold at auction and shipped to {destination_country}.
Car details:
Title: {car_title}
Year: {year}
Make: {make}
Model: {model}
Odometer: {mileage}
Condition notes: {condition_notes}

Provide:
1) A JSON object with fields: repair_cost_{auction_ccy}, resale_price_{dest_ccy}, repair_breakdown (list of estimated major repairs and costs in {auction_ccy}).
2) Conservative, median and optimistic resale estimates in {dest_ccy}.
3) Short rationale for country-specific cost multipliers for {destination_country}.

Respond ONLY with JSON.
"""
        ai_text = ask_openai_for_estimates(openai_key, prompt)
        if ai_text:
            try:
                # attempt to find JSON blob in reply
                start = ai_text.find("{")
                end = ai_text.rfind("}") + 1
                ai_json = json.loads(ai_text[start:end])
                if f"repair_cost_{auction_ccy}" in ai_json:
                    repair_cost_auction = float(ai_json[f"repair_cost_{auction_ccy}"])
                if f"resale_price_{dest_ccy}" in ai_json:
                    resale_price_dest_ccy = float(ai_json[f"resale_price_{dest_ccy}"])
                openai_notes = ai_json.get("notes", ai_text)
            except Exception as e:
                st.warning("OpenAI returned a response but parsing JSON failed; using manual inputs. Response excerpt shown.")
                st.text(ai_text[:1000])

    # Aggregate logistics & fees: convert destination currency items into auction currency
    import_clearance_auction = import_clearance * fx_rate_dest_to_auction
    local_trucking_auction = local_trucking * fx_rate_dest_to_auction
    local_sale_fees_auction = local_sale_fees * fx_rate_dest_to_auction

    logistics_total_auction = trucking_to_port + shipping_cost + local_trucking_auction
    other_fees_auction = copart_fees + import_clearance_auction + local_sale_fees_auction

    # Compute max bid
    max_bid = compute_max_bid(
        resale_price_dest_ccy=resale_price_dest_ccy,
        resale_ccy_to_auction_rate=fx_rate_dest_to_auction,
        repair_cost_auction_ccy=repair_cost_auction,
        logistics_cost_auction_ccy=logistics_total_auction,
        other_fees_auction_ccy=other_fees_auction,
        desired_profit_auction_ccy=desired_profit_auction
    )

    st.subheader("Core numbers")
    col1, col2 = st.columns(2)
    col1.metric("Expected resale (dest currency)", f"{resale_price_dest_ccy:,.2f} {dest_ccy}")
    col1.metric("Resale → Auction FX", f"1 {dest_ccy} = {fx_rate_dest_to_auction:.4f} {auction_ccy}")
    col2.metric("Repair estimate (auction ccy)", f"{repair_cost_auction:,.2f} {auction_ccy}")
    col2.metric("Total logistics (auction ccy)", f"{logistics_total_auction:,.2f} {auction_ccy}")

    st.markdown("### Final recommended bid")
    st.write(f"**Maximum recommended bid (in {auction_ccy}):**  {max_bid:,.2f} {auction_ccy}")
    st.write("This is the gross amount to bid at auction. Expect additional incremental fees/adjustments at sale.")

    st.markdown("### Cost breakdown (auction currency)")
    breakdown = {
        "Resale_in_auction_ccy": resale_price_dest_ccy * fx_rate_dest_to_auction,
        "Repair": repair_cost_auction,
        "Trucking_to_port": trucking_to_port,
        "Shipping": shipping_cost,
        "Copart_fees": copart_fees,
        "Import_clearance_in_auction": import_clearance_auction,
        "Local_trucking_in_auction": local_trucking_auction,
        "Local_sale_fees_in_auction": local_sale_fees_auction,
        "Desired_profit": desired_profit_auction
    }
    df_break = pd.DataFrame([breakdown]).T.reset_index()
    df_break.columns = ["item", "amount"]
    st.dataframe(df_break.style.format({"amount":"{:.2f}"}), height=300)

    # Sensitivity analysis
    st.markdown("### Sensitivity analysis")
    sens_cols = st.columns(3)
    # +-10% repair
    rp_low = compute_max_bid(resale_price_dest_ccy, fx_rate_dest_to_auction, repair_cost_auction*0.9, logistics_total_auction, other_fees_auction, desired_profit_auction)
    rp_high = compute_max_bid(resale_price_dest_ccy, fx_rate_dest_to_auction, repair_cost_auction*1.1, logistics_total_auction, other_fees_auction, desired_profit_auction)
    sens_cols[0].write(f"Repair cost ↓10% => max bid: {rp_low:,.2f} {auction_ccy}")
    sens_cols[0].write(f"Repair cost ↑10% => max bid: {rp_high:,.2f} {auction_ccy}")

    # +-10% FX (resale converted)
    fx_low = compute_max_bid(resale_price_dest_ccy, fx_rate_dest_to_auction*0.9, repair_cost_auction, logistics_total_auction, other_fees_auction, desired_profit_auction)
    fx_high = compute_max_bid(resale_price_dest_ccy, fx_rate_dest_to_auction*1.1, repair_cost_auction, logistics_total_auction, other_fees_auction, desired_profit_auction)
    sens_cols[1].write(f"FX ↓10% => max bid: {fx_low:,.2f} {auction_ccy}")
    sens_cols[1].write(f"FX ↑10% => max bid: {fx_high:,.2f} {auction_ccy}")

    # +-10% resale price
    rs_low = compute_max_bid(resale_price_dest_ccy*0.9, fx_rate_dest_to_auction, repair_cost_auction, logistics_total_auction, other_fees_auction, desired_profit_auction)
    rs_high = compute_max_bid(resale_price_dest_ccy*1.1, fx_rate_dest_to_auction, repair_cost_auction, logistics_total_auction, other_fees_auction, desired_profit_auction)
    sens_cols[2].write(f"Resale ↓10% => max bid: {rs_low:,.2f} {auction_ccy}")
    sens_cols[2].write(f"Resale ↑10% => max bid: {rs_high:,.2f} {auction_ccy}")

    # Downloadable summary
    st.markdown("### Download / Copy results")
    summary = {
        "copart_url": copart_url,
        "origin_country": origin_country,
        "destination_country": destination_country,
        "year": year, "make": make, "model": model,
        "resale_price_dest_ccy": resale_price_dest_ccy,
        "dest_ccy": dest_ccy,
        "fx_dest_to_auction": fx_rate_dest_to_auction,
        "repair_cost_auction": repair_cost_auction,
        "logistics_total_auction": logistics_total_auction,
        "other_fees_auction": other_fees_auction,
        "desired_profit_auction": desired_profit_auction,
        "recommended_max_bid_auction": max_bid
    }
    st.download_button("Download CSV summary", data=pd.DataFrame([summary]).to_csv(index=False), file_name="copart_bid_summary.csv", mime="text/csv")
    st.code(json.dumps(summary, indent=2))

    # Show OpenAI notes (if any)
    if openai_notes:
        st.markdown("### OpenAI estimates (raw)")
        st.text(openai_notes[:4000])

    st.markdown("### Notes, Limitations & Next steps")
    st.write("""
    - Copart pages are dynamic and may block scraping — the app offers manual inputs for reliability.
    - OpenAI integration is optional: it can provide estimated repair/resale numbers but verify and use manual overrides.
    - Exchange rate service (exchangerate.host) is free and may have occasional outages — manual FX override is recommended for critical decisions.
    - This tool gives an *estimate* only. Always include contingency buffer and perform local inspections/valuations before bidding.
    """)

