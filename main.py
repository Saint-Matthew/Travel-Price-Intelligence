import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def train_model():

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    data = pd.read_csv("dataset.csv")

    # ---------------------------------------------------
    # CLEAN DATA
    # ---------------------------------------------------

    data["destination"] = (
        data["destination"]
        .astype(str)
        .str.strip()
    )

    data["month"] = (
        data["month"]
        .astype(str)
        .str.strip()
    )

    data["airline"] = (
        data["airline"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------
    # LABEL ENCODERS
    # ---------------------------------------------------

    destination_encoder = LabelEncoder()
    month_encoder = LabelEncoder()
    airline_encoder = LabelEncoder()

    data["destination_encoded"] = (
        destination_encoder.fit_transform(
            data["destination"]
        )
    )

    data["month_encoded"] = (
        month_encoder.fit_transform(
            data["month"]
        )
    )

    data["airline_encoded"] = (
        airline_encoder.fit_transform(
            data["airline"]
        )
    )

    # ---------------------------------------------------
    # FEATURES
    # ---------------------------------------------------

    X = data[
        [
            "destination_encoded",
            "month_encoded",
            "airline_encoded",
            "booking_window_weeks"
        ]
    ]

    y = data["average_price_naira"]

    # ---------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = r2_score(y_test, predictions)

    return (
        model,
        accuracy,
        destination_encoder,
        month_encoder,
        airline_encoder
    )


def predict_price(
    model,
    destination_encoder,
    month_encoder,
    airline_encoder,
    destination,
    month,
    airline,
    booking_window
):

    destination_value = (
        destination_encoder.transform(
            [destination]
        )[0]
    )

    month_value = (
        month_encoder.transform(
            [month]
        )[0]
    )

    airline_value = (
        airline_encoder.transform(
            [airline]
        )[0]
    )

    prediction = model.predict(
        [[
            destination_value,
            month_value,
            airline_value,
            booking_window
        ]]
    )

    return prediction[0]

