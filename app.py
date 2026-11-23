import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Purchase Order Recommendation System",
    page_icon="📦",
    layout="wide",
)

TEXTS = {
    "ar": {
        "dir": "rtl",
        "align": "right",
        "lang_label": "اللغة / Language",
        "title": "📦 نظام توصية أوامر الشراء (PO)",
        "caption": "يقترح كميات إعادة الطلب بناءً على سجل الشراء والاستهلاك التاريخي",
        "sidebar_header": "⚙️ إعدادات السياسة الشرائية",
        "coverage_days": "أيام التغطية المستهدفة بعد الاستلام",
        "service_level": "مستوى الخدمة (معامل Z)",
        "z_90": "مستوى خدمة 90% (z = 1.28)",
        "z_95": "مستوى خدمة 95% (z = 1.65)",
        "z_98": "مستوى خدمة 98% (z = 2.05)",
        "z_99": "مستوى خدمة 99% (z = 2.33)",
        "use_budget": "تطبيق حد أقصى لميزانية أوامر الشراء",
        "budget_limit": "أقصى ميزانية إجمالية ($)",
        "upload_label": "رفع ملف CSV لسجل الشراء أو الاستهلاك (اختياري)",
        "sample_info": "يتم عرض بيانات تجريبية مدمجة (60 يوماً عبر 4 منتجات). يمكنك رفع ملف CSV خاص بك.",
        "missing_cols": "أعمدة مطلوبة مفقودة في ملف CSV: ",
        "items_needing_po": "المنتجات المحتاجة لطلب شراء",
        "total_units": "إجمالي الوحدات المطلوبة",
        "est_total_cost": "التكلفة التقديرية الإجمالية",
        "suggested_pos": "📋 أوامر الشراء المقترحة",
        "download_csv": "⬇️ تحميل أوامر الشراء المقترحة (CSV)",
        "raw_data_exp": "عرض البيانات التاريخية الخام",
        "order_rec": "يُوصى بالطلب",
        "suff_stock": "المخزون كافٍ",
        "skip_budget": "تم التجاوز (حد الميزانية)",
        "adj_budget": "تم التعديل (حد الميزانية)",
        "cols": {
            "SKU": "رمز المنتج (SKU)",
            "Product": "اسم المنتج",
            "Supplier": "المورد",
            "Current Stock": "المخزون الحالي",
            "Avg Daily Demand": "متوسط الطلب اليومي",
            "Days of Cover": "أيام التغطية المتبقية",
            "Lead Time (d)": "مدة التوريد (أيام)",
            "Safety Stock": "مخزون الأمان",
            "Reorder Point (ROP)": "نقطة إعادة الطلب (ROP)",
            "MOQ": "الحد الأدنى للطلب (MOQ)",
            "Recommended PO Qty": "الكمية الموصى بطلبها",
            "Unit Cost": "تكلفة الوحدة",
            "Estimated PO Cost": "التكلفة التقديرية",
            "Status": "الحالة",
        },
        "products": {
            "SKU-001": ("ملصقات حرارية (رول)", "إمدادات ألفا"),
            "SKU-002": ("صناديق كرتونية متوسطة", "باك رايت"),
            "SKU-003": ("لفائف تغليف مطاطية", "إمدادات ألفا"),
            "SKU-004": ("شريط طابعة باركود", "أوبتيتك للموردين"),
        },
    },
    "en": {
        "dir": "ltr",
        "align": "left",
        "lang_label": "Language / اللغة",
        "title": "📦 Purchase Order (PO) Recommendation Engine",
        "caption": "Suggests reorder quantities from historical purchasing & usage data",
        "sidebar_header": "⚙️ Policy Parameters",
        "coverage_days": "Target Coverage After Delivery (days)",
        "service_level": "Service Level (Z-score)",
        "z_90": "90% Service Level (z = 1.28)",
        "z_95": "95% Service Level (z = 1.65)",
        "z_98": "98% Service Level (z = 2.05)",
        "z_99": "99% Service Level (z = 2.33)",
        "use_budget": "Apply Maximum PO Budget",
        "budget_limit": "Max Total Budget ($)",
        "upload_label": "Upload historical purchasing/usage CSV (optional)",
        "sample_info": "Showing built-in sample historical data (60 days across 4 SKUs). Upload a CSV to use your own.",
        "missing_cols": "Missing required columns in CSV: ",
        "items_needing_po": "Items Needing PO",
        "total_units": "Total Units to Order",
        "est_total_cost": "Estimated Total PO Cost",
        "suggested_pos": "📋 Suggested Purchase Orders",
        "download_csv": "⬇️ Download Recommended POs (CSV)",
        "raw_data_exp": "View Raw Historical Data",
        "order_rec": "Order Recommended",
        "suff_stock": "Sufficient Stock",
        "skip_budget": "Skipped (Budget Cap)",
        "adj_budget": "Adjusted (Budget Cap)",
        "cols": {
            "SKU": "SKU",
            "Product": "Product",
            "Supplier": "Supplier",
            "Current Stock": "Current Stock",
            "Avg Daily Demand": "Avg Daily Demand",
            "Days of Cover": "Days of Cover",
            "Lead Time (d)": "Lead Time (d)",
            "Safety Stock": "Safety Stock",
            "Reorder Point (ROP)": "Reorder Point (ROP)",
            "MOQ": "MOQ",
            "Recommended PO Qty": "Recommended PO Qty",
            "Unit Cost": "Unit Cost",
            "Estimated PO Cost": "Estimated PO Cost",
            "Status": "Status",
        },
        "products": {
            "SKU-001": ("Thermal Labels (Roll)", "Alpha Supplies"),
            "SKU-002": ("Corrugated Boxes Medium", "PackRight Co."),
            "SKU-003": ("Stretch Wrap Film", "Alpha Supplies"),
            "SKU-004": ("Barcode Scanner Ribbon", "OptiTech Vendors"),
        },
    },
}


def default_sample_data(lang: str) -> pd.DataFrame:
    """Create sample historical purchase/usage records over recent days."""
    rng = np.random.default_rng(42)
    names_map = TEXTS[lang]["products"]
    products = [
        {
            "sku": "SKU-001",
            "product_name": names_map["SKU-001"][0],
            "supplier": names_map["SKU-001"][1],
            "current_stock": 45,
            "lead_time_days": 5,
            "unit_cost": 4.50,
            "moq": 30,
            "order_multiple": 10,
            "base_daily_qty": 12,
        },
        {
            "sku": "SKU-002",
            "product_name": names_map["SKU-002"][0],
            "supplier": names_map["SKU-002"][1],
            "current_stock": 190,
            "lead_time_days": 7,
            "unit_cost": 1.25,
            "moq": 100,
            "order_multiple": 50,
            "base_daily_qty": 25,
        },
        {
            "sku": "SKU-003",
            "product_name": names_map["SKU-003"][0],
            "supplier": names_map["SKU-003"][1],
            "current_stock": 18,
            "lead_time_days": 4,
            "unit_cost": 14.00,
            "moq": 12,
            "order_multiple": 6,
            "base_daily_qty": 4,
        },
        {
            "sku": "SKU-004",
            "product_name": names_map["SKU-004"][0],
            "supplier": names_map["SKU-004"][1],
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
    t: dict,
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
            status = t["order_rec"]
        else:
            rec_qty = 0
            status = t["suff_stock"]

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
                result.at[idx, "Status"] = t["skip_budget"]
            elif cost * qty <= remaining:
                spent += cost * qty
            else:
                affordable_units = int(remaining // cost)
                step = max(1, int(row["MOQ"]))
                mult_val = max(1, int(df.loc[df["sku"] == row["SKU"], "order_multiple"].iloc[0]))
                unit_step = max(step, mult_val)
                adjusted_qty = (affordable_units // unit_step) * unit_step
                if adjusted_qty >= step:
                    result.at[idx, "Recommended PO Qty"] = adjusted_qty
                    result.at[idx, "Estimated PO Cost"] = round(adjusted_qty * cost, 2)
                    result.at[idx, "Status"] = t["adj_budget"]
                    spent += adjusted_qty * cost
                else:
                    result.at[idx, "Recommended PO Qty"] = 0
                    result.at[idx, "Estimated PO Cost"] = 0.0
                    result.at[idx, "Status"] = t["skip_budget"]

    return result.sort_values(by=["Recommended PO Qty", "SKU"], ascending=[False, True]).reset_index(drop=True)


with st.sidebar:
    lang = st.selectbox(
        "اللغة / Language",
        options=["ar", "en"],
        index=0,
        format_func=lambda x: "العربية" if x == "ar" else "English",
    )

t = TEXTS[lang]

st.markdown(
    f"""
    <div dir="{t['dir']}" style="text-align: {t['align']};">
        <h1>{t['title']}</h1>
        <p>{t['caption']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(t["sidebar_header"])
    coverage_days = st.slider(t["coverage_days"], min_value=3, max_value=60, value=14)
    service_level_label = st.selectbox(
        t["service_level"],
        options=[
            (t["z_90"], 1.28),
            (t["z_95"], 1.65),
            (t["z_98"], 2.05),
            (t["z_99"], 2.33),
        ],
        index=1,
        format_func=lambda item: item[0],
    )
    z_val = service_level_label[1]

    use_budget = st.checkbox(t["use_budget"], value=False)
    budget_limit = st.number_input(t["budget_limit"], min_value=0.0, value=1500.0, step=50.0) if use_budget else None

uploaded_file = st.file_uploader(t["upload_label"], type=["csv"])

if uploaded_file is not None:
    hist_df = pd.read_csv(uploaded_file)
else:
    hist_df = default_sample_data(lang)
    st.info(t["sample_info"])

required_cols = {"date", "sku", "product_name", "supplier", "quantity_used", "unit_cost", "current_stock", "lead_time_days"}
missing = required_cols - set(hist_df.columns)
if missing:
    st.error(t["missing_cols"] + ", ".join(sorted(missing)))
    st.stop()

if "moq" not in hist_df.columns:
    hist_df["moq"] = 1
if "order_multiple" not in hist_df.columns:
    hist_df["order_multiple"] = 1

po_df = compute_po_recommendations(
    hist_df,
    t=t,
    target_coverage_days=coverage_days,
    z_score=z_val,
    max_budget=budget_limit,
)

active_pos = po_df[po_df["Recommended PO Qty"] > 0]

col1, col2, col3 = st.columns(3)
col1.metric(t["items_needing_po"], len(active_pos))
col2.metric(t["total_units"], int(active_pos["Recommended PO Qty"].sum()) if not active_pos.empty else 0)
col3.metric(t["est_total_cost"], f"${active_pos['Estimated PO Cost'].sum():,.2f}" if not active_pos.empty else "$0.00")

st.subheader(t["suggested_pos"])
display_po_df = po_df.rename(columns=t["cols"])
st.dataframe(display_po_df, use_container_width=True, hide_index=True)

csv_buffer = io.StringIO()
display_po_df[display_po_df[t["cols"]["Recommended PO Qty"]] > 0].to_csv(csv_buffer, index=False)
st.download_button(
    label=t["download_csv"],
    data=csv_buffer.getvalue(),
    file_name="recommended_purchase_orders.csv",
    mime="text/csv",
)

with st.expander(t["raw_data_exp"]):
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
