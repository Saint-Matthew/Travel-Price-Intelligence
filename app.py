import streamlit as st
import pandas as pd
import plotly.express as px

from main import train_model, predict_price

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Travel Price Intelligence Dashboard",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("✈️ Nigerian Travel Price Intelligence Dashboard")

st.write(
    """
    Analyze international travel prices for Nigerian travelers,
    compare airlines, explore destinations, and predict
    estimated flight costs using machine learning.
    """
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

data = pd.read_csv("dataset.csv")

(
    model,
    accuracy,
    destination_encoder,
    month_encoder,
    airline_encoder
) = train_model()

# ---------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------

st.metric(
    label="Model Accuracy (R²)",
    value=f"{accuracy:.2f}"
)

st.divider()

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.subheader("📊 Dataset Overview")

st.dataframe(
    data,
    use_container_width=True,
    height=300
)

st.divider()

# ---------------------------------------------------
# PREDICTION ENGINE
# ---------------------------------------------------

st.subheader("🌍 Travel Prediction Engine")

col1, col2 = st.columns(2)

with col1:

    destination = st.selectbox(
        "Select Destination Country",
        sorted(data["destination"].unique())
    )

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    month = st.selectbox(
        "Select Travel Month",
        month_order
    )

with col2:

    airline = st.selectbox(
        "Select Airline",
        sorted(data["airline"].unique())
    )

    booking_window = st.slider(
        "Weeks Before Travel",
        1,
        12,
        6
    )

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict Travel Cost"):

    predicted_price = predict_price(
        model,
        destination_encoder,
        month_encoder,
        airline_encoder,
        destination,
        month,
        airline,
        booking_window
    )

    st.success(
        f"Estimated Flight Cost: ₦{predicted_price:,.0f}"
    )

    st.subheader("🧠 Smart Insights")

    if month in ["July", "August", "December"]:
        st.info(
            "Peak travel periods generally experience higher prices."
        )

    if booking_window <= 3:
        st.warning(
            "Late bookings tend to increase flight prices significantly."
        )

    if destination in [
        "USA",
        "Canada",
        "United Kingdom",
        "France"
    ]:
        st.info(
            "Long-haul international routes usually require larger travel budgets."
        )

st.divider()

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------

st.sidebar.header("Dashboard Filters")

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    sorted(data["destination"].unique()),
    default=[
        "USA",
        "United Kingdom",
        "Canada"
    ]
)

selected_airlines = st.sidebar.multiselect(
    "Select Airlines",
    sorted(data["airline"].unique()),
    default=[
        "Emirates",
        "Qatar Airways",
        "British Airways"
    ]
)

filtered_data = data[
    (data["destination"].isin(selected_countries)) &
    (data["airline"].isin(selected_airlines))
]

# ---------------------------------------------------
# KPI METRICS
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Countries",
        len(filtered_data["destination"].unique())
    )

with col2:
    st.metric(
        "Airlines",
        len(filtered_data["airline"].unique())
    )

with col3:
    st.metric(
        "Average Ticket Price",
        f"₦{int(filtered_data['average_price_naira'].mean()):,}"
    )

st.divider()

# ---------------------------------------------------
# ANALYTICS SECTION
# ---------------------------------------------------

st.subheader("📈 Interactive Travel Analytics")

tab1, tab2, tab3 = st.tabs([
    "🌍 Destinations",
    "📈 Trends",
    "✈️ Airlines"
])

# ---------------------------------------------------
# DESTINATION ANALYTICS
# ---------------------------------------------------

top_destinations = (
    filtered_data.groupby("destination")[
        "average_price_naira"
    ]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig1 = px.bar(
    top_destinations,
    x="destination",
    y="average_price_naira",
    color="average_price_naira",
    text_auto=".2s",
    title="Top Destination Flight Prices"
)

fig1.update_layout(
    template="plotly_dark",
    xaxis_title="Destination",
    yaxis_title="Average Price (₦)"
)

# ---------------------------------------------------
# MONTHLY TRENDS
# ---------------------------------------------------

monthly_trends = (
    filtered_data.groupby("month")[
        "average_price_naira"
    ]
    .mean()
    .reset_index()
)

monthly_trends["month"] = pd.Categorical(
    monthly_trends["month"],
    categories=month_order,
    ordered=True
)

monthly_trends = monthly_trends.sort_values(
    "month"
)

fig2 = px.line(
    monthly_trends,
    x="month",
    y="average_price_naira",
    markers=True,
    title="Average Monthly Flight Prices"
)

fig2.update_layout(
    template="plotly_dark",
    xaxis_title="Month",
    yaxis_title="Average Price (₦)"
)

# ---------------------------------------------------
# AIRLINE ANALYTICS
# ---------------------------------------------------

airline_prices = (
    filtered_data.groupby("airline")[
        "average_price_naira"
    ]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig3 = px.bar(
    airline_prices,
    x="airline",
    y="average_price_naira",
    color="average_price_naira",
    title="Average Airline Pricing"
)

fig3.update_layout(
    template="plotly_dark",
    xaxis_title="Airline",
    yaxis_title="Average Price (₦)"
)

# ---------------------------------------------------
# TABS DISPLAY
# ---------------------------------------------------

with tab1:
    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with tab2:
    st.plotly_chart(
        fig2,
        use_container_width=True
    )

with tab3:
    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# ---------------------------------------------------
# POPULAR DESTINATIONS
# ---------------------------------------------------

st.divider()

st.subheader("🔥 Most Popular Destinations")

popular_destinations = {
    "USA": ["New York", "Texas", "Atlanta"],
    "Canada": ["Toronto", "Vancouver", "Ottawa"],
    "United Kingdom": ["London", "Manchester", "Birmingham"],
    "France": ["Paris", "Lyon", "Marseille"],
    "South Africa": ["Cape Town", "Johannesburg", "Durban"]
}

selected_country = st.selectbox(
    "Select Country to Explore Cities",
    list(popular_destinations.keys())
)

st.write(
    popular_destinations[selected_country]
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit, Plotly, Pandas, and Scikit-learn"
)

