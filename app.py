import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Aircraft Fuel Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2.0rem !important;
    padding-bottom: 2.5rem !important;
}

/* Prevent the Streamlit toolbar from covering content */
[data-testid="stToolbar"] {
    z-index: 999999;
}

[data-testid="stHeader"] {
    height: 3.5rem;
}

.main .block-container {
    padding-top: 5rem !important;
}


.dashboard-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

.section-title {
    font-size: 22px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 15px;
}

.prediction-box {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.prediction-value {
    font-size: 42px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    fuel = pd.read_parquet("fuel_train.parquet")
    flights = pd.read_parquet("flightlist_train.parquet")

    # Total fuel consumed by each flight
    fuel_total = (
        fuel.groupby("flight_id", as_index=False)["fuel_kg"]
        .sum()
        .rename(columns={"fuel_kg": "total_fuel_kg"})
    )

    # Merge flight information and fuel information
    df = flights.merge(
        fuel_total,
        on="flight_id",
        how="inner"
    )

    # Datetime conversion
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df["takeoff"] = pd.to_datetime(df["takeoff"])
    df["landed"] = pd.to_datetime(df["landed"])

    # Flight duration
    df["flight_duration_hours"] = (
        df["landed"] - df["takeoff"]
    ).dt.total_seconds() / 3600

    # Feature engineering
    df["month"] = df["flight_date"].dt.month
    df["day_of_week"] = df["flight_date"].dt.dayofweek

    df["route"] = (
        df["origin_icao"] +
        " → " +
        df["destination_icao"]
    )

    return df


@st.cache_resource
def load_model():

    model = joblib.load(
        "aircraft_fuel_random_forest_compressed.pkl"
    )

    preprocessor = joblib.load(
        "aircraft_fuel_preprocessor_compressed.pkl"
    )

    return model, preprocessor


df = load_data()
model, preprocessor = load_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("✈️ Aircraft Fuel Analytics")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "📊 Executive Dashboard",
        "✈️ Aircraft & Flight Analysis",
        "🛣️ Route & Fuel Analysis",
        "🤖 Fuel Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning Based Aircraft "
    "Flight Fuel Consumption Prediction"
)


# =========================================================
# PAGE 1
# EXECUTIVE DASHBOARD
# =========================================================

if page == "📊 Executive Dashboard":

    st.markdown(
        '<div class="dashboard-title">'
        'Aircraft Flight Fuel Analytics Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Interactive analysis of aircraft flight operations '
        'and fuel consumption'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------
    # KPI CARDS
    # -----------------------------

    total_flights = len(df)
    avg_fuel = df["total_fuel_kg"].mean()
    avg_duration = df["flight_duration_hours"].mean()
    max_fuel = df["total_fuel_kg"].max()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Flights",
        f"{total_flights:,}"
    )

    col2.metric(
        "Average Fuel",
        f"{avg_fuel:,.0f} kg"
    )

    col3.metric(
        "Average Duration",
        f"{avg_duration:.2f} hrs"
    )

    col4.metric(
        "Maximum Fuel",
        f"{max_fuel:,.0f} kg"
    )

    st.markdown("---")

    # -----------------------------
    # FILTERS
    # -----------------------------

    st.markdown(
        '<div class="section-title">Dashboard Filters</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    aircraft_filter = c1.multiselect(
        "Aircraft Type",
        sorted(df["aircraft_type"].dropna().unique())
    )

    origin_filter = c2.multiselect(
        "Origin Airport",
        sorted(df["origin_icao"].dropna().unique())
    )

    destination_filter = c3.multiselect(
        "Destination Airport",
        sorted(df["destination_icao"].dropna().unique())
    )

    filtered_df = df.copy()

    if aircraft_filter:
        filtered_df = filtered_df[
            filtered_df["aircraft_type"].isin(
                aircraft_filter
            )
        ]

    if origin_filter:
        filtered_df = filtered_df[
            filtered_df["origin_icao"].isin(
                origin_filter
            )
        ]

    if destination_filter:
        filtered_df = filtered_df[
            filtered_df["destination_icao"].isin(
                destination_filter
            )
        ]

    # -----------------------------
    # CHARTS
    # -----------------------------

    c1, c2 = st.columns(2)

    with c1:

        monthly = (
            filtered_df
            .groupby("month")["total_fuel_kg"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="month",
            y="total_fuel_kg",
            markers=True,
            title="Average Fuel Consumption by Month"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Fuel (kg)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        aircraft_fuel = (
            filtered_df
            .groupby("aircraft_type")
            ["total_fuel_kg"]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            aircraft_fuel,
            x="total_fuel_kg",
            y="aircraft_type",
            orientation="h",
            title="Top Aircraft Types by Average Fuel"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------
    # DISTRIBUTION
    # -----------------------------

    fig = px.histogram(
        filtered_df,
        x="total_fuel_kg",
        nbins=50,
        title="Fuel Consumption Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PAGE 2
# AIRCRAFT & FLIGHT ANALYSIS
# =========================================================

elif page == "✈️ Aircraft & Flight Analysis":

    st.markdown(
        '<div class="dashboard-title">'
        'Aircraft & Flight Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Explore aircraft performance, flight duration "
        "and fuel consumption."
    )

    selected_aircraft = st.selectbox(
        "Select Aircraft Type",
        sorted(df["aircraft_type"].unique())
    )

    aircraft_df = df[
        df["aircraft_type"] == selected_aircraft
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Flights",
        f"{len(aircraft_df):,}"
    )

    c2.metric(
        "Average Fuel",
        f"{aircraft_df['total_fuel_kg'].mean():,.0f} kg"
    )

    c3.metric(
        "Average Duration",
        f"{aircraft_df['flight_duration_hours'].mean():.2f} hrs"
    )

    st.markdown("---")

    # Duration vs Fuel

    fig = px.scatter(
        aircraft_df,
        x="flight_duration_hours",
        y="total_fuel_kg",
        hover_data=[
            "origin_icao",
            "destination_icao"
        ],
        title=f"{selected_aircraft}: Flight Duration vs Fuel"
    )

    fig.update_layout(
        xaxis_title="Flight Duration (hours)",
        yaxis_title="Fuel Consumption (kg)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Aircraft comparison

    comparison = (
        df.groupby("aircraft_type")
        .agg(
            Flights=("flight_id", "count"),
            Average_Fuel=("total_fuel_kg", "mean"),
            Average_Duration=("flight_duration_hours", "mean")
        )
        .sort_values(
            "Average_Fuel",
            ascending=False
        )
        .reset_index()
    )

    st.subheader(
        "Aircraft Performance Comparison"
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )


# =========================================================
# PAGE 3
# ROUTE & FUEL ANALYSIS
# =========================================================

elif page == "🛣️ Route & Fuel Analysis":

    st.markdown(
        '<div class="dashboard-title">'
        'Route & Fuel Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Analyze fuel consumption across routes "
        "and flight durations."
    )

    # Top routes

    route_summary = (
        df.groupby("route")
        .agg(
            Flights=("flight_id", "count"),
            Average_Fuel=("total_fuel_kg", "mean"),
            Average_Duration=("flight_duration_hours", "mean")
        )
        .reset_index()
    )

    top_routes = (
        route_summary
        .sort_values(
            "Average_Fuel",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        top_routes,
        x="Average_Fuel",
        y="route",
        orientation="h",
        hover_data=[
            "Flights",
            "Average_Duration"
        ],
        title="Top 15 Routes by Average Fuel Consumption"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Scatter analysis

    fig = px.scatter(
        df,
        x="flight_duration_hours",
        y="total_fuel_kg",
        color="aircraft_type",
        hover_data=[
            "origin_icao",
            "destination_icao"
        ],
        title="Flight Duration vs Fuel Consumption"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Correlation

    correlation = df[
        [
            "flight_duration_hours",
            "total_fuel_kg"
        ]
    ].corr().iloc[0, 1]

    st.metric(
        "Duration–Fuel Correlation",
        f"{correlation:.3f}"
    )


# =========================================================
# PAGE 4
# MACHINE LEARNING PREDICTION
# =========================================================

elif page == "🤖 Fuel Prediction":

    st.markdown(
        '<div class="dashboard-title">'
        'AI Fuel Consumption Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Enter flight characteristics to estimate "
        "total aircraft fuel consumption."
    )

    st.markdown("---")

    # Input columns

    c1, c2 = st.columns(2)

    with c1:

        aircraft_type = st.selectbox(
            "Aircraft Type",
            sorted(
                df["aircraft_type"]
                .dropna()
                .unique()
            )
        )

        origin = st.selectbox(
            "Origin ICAO",
            sorted(
                df["origin_icao"]
                .dropna()
                .unique()
            )
        )

        flight_duration = st.number_input(
            "Flight Duration (hours)",
            min_value=0.1,
            max_value=20.0,
            value=4.0,
            step=0.1
        )

    with c2:

        destination = st.selectbox(
            "Destination ICAO",
            sorted(
                df["destination_icao"]
                .dropna()
                .unique()
            )
        )

        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=0
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        day_number = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ].index(day_of_week)

    st.markdown("---")

    # Prediction button

    if st.button(
        "🚀 Predict Fuel Consumption",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "aircraft_type": [aircraft_type],
            "flight_duration_hours": [
                flight_duration
            ],
            "origin_icao": [origin],
            "destination_icao": [
                destination
            ],
            "month": [month],
            "day_of_week": [day_number]
        })

        processed_input = (
            preprocessor.transform(
                input_data
            )
        )

        prediction = model.predict(
            processed_input
        )[0]

        st.markdown(
            f"""
            <div class="prediction-box">
                <div>Estimated Fuel Consumption</div>
                <div class="prediction-value">
                    {prediction:,.2f} kg
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "Prediction generated successfully "
            "using the trained Random Forest model."
        )

        st.markdown("### Flight Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "Aircraft Type",
                "Origin",
                "Destination",
                "Flight Duration",
                "Month",
                "Day"
            ],
            "Value": [
                aircraft_type,
                origin,
                destination,
                f"{flight_duration:.2f} hours",
                month,
                day_of_week
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    # Model performance

    st.markdown("---")

    st.subheader(
        "Machine Learning Model Performance"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        "2,102.23 kg"
    )

    c2.metric(
        "RMSE",
        "4,232.53 kg"
    )

    c3.metric(
        "R² Score",
        "0.7437"
    )

    st.info(
        "The final Random Forest model explains approximately "
        "74.37% of the variation in fuel consumption on the "
        "held-out test dataset."
    )
