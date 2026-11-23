import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Purchase Order Recommendation System",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Purchase Order (PO) Recommendation Engine")
st.caption("Suggests reorder quantities from historical purchasing & usage data")


def default_sample_data() -> pd.DataFrame:
    """Create sample historical purchase/usage records over recent days."""
    rng = np.random.default_rng(42)
    products = [
        {
            "sku": "SKU-001",
            "product_name": "Thermal Labels (Roll)",
            "supplier": "Alpha Supplies",
            "current_stock": 45,
            "lead_time_days": 5,
            "unit_cost": 4.50,
            "moq": 30,
            "order_multiple": 10,
            "base_daily_qty": 12,
        },
        {
            "sku": "SKU-002",
            "product_name": "Corrugated Boxes Medium",
            "supplier": "PackRight Co.",
            "current_stock": 190,
            "lead_time_days": 7,
            "unit_cost": 1.25,
            "moq": 100,
            "order_multiple": 50,
            "base_daily_qty": 25,
        },
        {
            "sku": "SKU-003",
            "product_name": "Stretch Wrap Film",
            "supplier": "Alpha Supplies",
            "current_stock": 18,
            "lead_time_days": 4,
            "unit_cost": 14.00,
            "moq": 12,
            "order_multiple": 6,
            "base_daily_qty": 4,
        },
        {
            "sku": "SKU-004",
            "product_name": "Barcode Scanner Ribbon",
            "supplier": "OptiTech Vendors",
            "current_stock": 85,
            "lead_time_days": 10,
            "unit_cost": 9.80,
            "moq": 20,
            "order_multiple": 5,
            "base_daily_qty": 5,
        },
    ]

    rows = []
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=60, freq="D")

    for p in products:
        for d in dates:
            qty = max(0, int(rng.poisson(lam=p["base_daily_qty"])))
            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "sku": p["sku"],
                    "product_name": p["product_name"],
                    "supplier": p["supplier"],
                    "quantity_used": qty,
                    "unit_cost": p["unit_cost"],
                    "current_stock": p["current_stock"],
                    "lead_time_days": p["lead_time_days"],
                    "moq": p["moq"],
                    "order_multiple": p["order_multiple"],
                }
            )
    return pd.DataFrame(rows)


def round_to_step(qty: float, moq: int, multiple: int) -> int:
    """Round quantity up to satisfy MOQ and order multiple increments."""
    if qty <= 0:
        return 0
    target = max(float(qty), float(moq))
    step = max(1, int(multiple))
    return int(np.ceil(target / step) * step)


def compute_po_recommendations(
    df: pd.DataFrame,
    target_coverage_days: int = 14,
    z_score: float = 1.65,
    max_budget: float | None = None,
) -> pd.DataFrame:
    """Aggregate historical data per SKU and recommend PO quantities."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["quantity_used"] = pd.to_numeric(work["quantity_used"], errors="coerce").fillna(0.0)
    work["unit_cost"] = pd.to_numeric(work["unit_cost"], errors="coerce").fillna(0.0)
    work["current_stock"] = pd.to_numeric(work["current_stock"], errors="coerce").fillna(0.0)
    work["lead_time_days"] = pd.to_numeric(work["lead_time_days"], errors="coerce").fillna(1.0)
    work["moq"] = pd.to_numeric(work["moq"], errors="coerce").fillna(1).astype(int)
    work["order_multiple"] = pd.to_numeric(work["order_multiple"], errors="coerce").fillna(1).astype(int)

    daily = (
        work.groupby(
            ["sku", "product_name", "supplier", "date", "unit_cost", "current_stock", "lead_time_days", "moq", "order_multiple"],
            as_index=False,
        )["quantity_used"]
        .sum()
    )

    records = []
    for (sku, name, supplier, unit_cost, stock, lead_time, moq, mult), grp in daily.groupby(
        ["sku", "product_name", "supplier", "unit_cost", "current_stock", "lead_time_days", "moq", "order_multiple"]
    ):
        avg_daily = float(grp["quantity_used"].mean())
        std_daily = float(grp["quantity_used"].std(ddof=1)) if len(grp) > 1 else 0.0

        safety_stock = z_score * std_daily * np.sqrt(max(lead_time, 0.0))
        lead_time_demand = avg_daily * lead_time
        rop = lead_time_demand + safety_stock

        desired_stock = avg_daily * (lead_time + target_coverage_days) + safety_stock
        net_shortfall = desired_stock - stock

        if stock <= rop and net_shortfall > 0:
            rec_qty = round_to_step(net_shortfall, moq=moq, multiple=mult)
            status = "Order Recommended"
        else:
            rec_qty = 0
            status = "Sufficient Stock"

        est_cost = round(rec_qty * unit_cost, 2)
        days_of_cover = round(stock / avg_daily, 1) if avg_daily > 0 else 999.0

        records.append(
            {
                "SKU": sku,
                "Product": name,
                "Supplier": supplier,
                "Current Stock": int(stock),
                "Avg Daily Demand": round(avg_daily, 2),
                "Days of Cover": days_of_cover,
                "Lead Time (d)": int(lead_time),
                "Safety Stock": round(safety_stock, 1),
                "Reorder Point (ROP)": round(rop, 1),
                "MOQ": moq,
                "Recommended PO Qty": rec_qty,
                "Unit Cost": round(unit_cost, 2),
                "Estimated PO Cost": est_cost,
                "Status": status,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    if max_budget is not None and max_budget >= 0:
        result = result.sort_values(by=["Recommended PO Qty", "Days of Cover"], ascending=[False, True]).reset_index(drop=True)
        spent = 0.0
        for idx, row in result.iterrows():
            qty = int(row["Recommended PO Qty"])
            cost = float(row["Unit Cost"])
            if qty <= 0:
                continue
            remaining = max_budget - spent
            if remaining <= 0 or cost <= 0:
                result.at[idx, "Recommended PO Qty"] = 0
                result.at[idx, "Estimated PO Cost"] = 0.0
                result.at[idx, "Status"] = "Skipped (Budget Cap)"
            elif cost * qty <= remaining:
                spent += cost * qty
            else:
                affordable_units = int(remaining // cost)
                step = max(1, int(row["MOQ"]))
                mult = max(1, int(df.loc[df["sku"] == row["SKU"], "order_multiple"].iloc[0]))
                unit_step = max(step, mult)
                adjusted_qty = (affordable_units // unit_step) * unit_step
                if adjusted_qty >= step:
                    result.at[idx, "Recommended PO Qty"] = adjusted_qty
                    result.at[idx, "Estimated PO Cost"] = round(adjusted_qty * cost, 2)
                    result.at[idx, "Status"] = "Adjusted (Budget Cap)"
                    spent += adjusted_qty * cost
                else:
                    result.at[idx, "Recommended PO Qty"] = 0
                    result.at[idx, "Estimated PO Cost"] = 0.0
                    result.at[idx, "Status"] = "Skipped (Budget Cap)"

    return result.sort_values(by=["Recommended PO Qty", "SKU"], ascending=[False, True]).reset_index(drop=True)


with st.sidebar:
    st.header("⚙️ Policy Parameters")
    coverage_days = st.slider("Target Coverage After Delivery (days)", min_value=3, max_value=60, value=14)
    service_level_label = st.selectbox(
        "Service Level (Z-score)",
        options=[
            ("90% Service Level (z = 1.28)", 1.28),
            ("95% Service Level (z = 1.65)", 1.65),
            ("98% Service Level (z = 2.05)", 2.05),
            ("99% Service Level (z = 2.33)", 2.33),
        ],
        index=1,
        format_func=lambda item: item[0],
    )
    z_val = service_level_label[1]

    use_budget = st.checkbox("Apply Maximum PO Budget", value=False)
    budget_limit = st.number_input("Max Total Budget ($)", min_value=0.0, value=1500.0, step=50.0) if use_budget else None

uploaded_file = st.file_uploader("Upload historical purchasing/usage CSV (optional)", type=["csv"])

if uploaded_file is not None:
    hist_df = pd.read_csv(uploaded_file)
else:
    hist_df = default_sample_data()
    st.info("Showing built-in sample historical data (60 days across 4 SKUs). Upload a CSV to use your own.")

required_cols = {"date", "sku", "product_name", "supplier", "quantity_used", "unit_cost", "current_stock", "lead_time_days"}
missing = required_cols - set(hist_df.columns)
if missing:
    st.error(f"Missing required columns in CSV: {', '.join(sorted(missing))}")
    st.stop()

if "moq" not in hist_df.columns:
    hist_df["moq"] = 1
if "order_multiple" not in hist_df.columns:
    hist_df["order_multiple"] = 1

po_df = compute_po_recommendations(
    hist_df,
    target_coverage_days=coverage_days,
    z_score=z_val,
    max_budget=budget_limit,
)

active_pos = po_df[po_df["Recommended PO Qty"] > 0]

col1, col2, col3 = st.columns(3)
col1.metric("Items Needing PO", len(active_pos))
col2.metric("Total Units to Order", int(active_pos["Recommended PO Qty"].sum()) if not active_pos.empty else 0)
col3.metric("Estimated Total PO Cost", f"${active_pos['Estimated PO Cost'].sum():,.2f}" if not active_pos.empty else "$0.00")

st.subheader("📋 Suggested Purchase Orders")
st.dataframe(po_df, use_container_width=True, hide_index=True)

csv_buffer = io.StringIO()
active_pos.to_csv(csv_buffer, index=False)
st.download_button(
    label="⬇️ Download Recommended POs (CSV)",
    data=csv_buffer.getvalue(),
    file_name="recommended_purchase_orders.csv",
    mime="text/csv",
)

with st.expander("View Raw Historical Data"):
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
