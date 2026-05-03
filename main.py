from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import pickle

app = FastAPI(
    title="F1 Ranker API",
    description="FastAPI for predicting Formula 1 league table rankings, most improved drivers, driver comparison and race result history",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("f1_xgb_ranker_tuned.pkl", "rb") as f:
    ranker = pickle.load(f)

label_encoders = joblib.load("ranker_label_encoders.pkl")
feature_columns = joblib.load("ranker_feature_columns.pkl")

data = pd.read_csv("f1_feature_engineered_dataset.csv", low_memory=False)

drivers_raw_df = pd.read_csv("drivers.csv", low_memory=False)
constructors_raw_df = pd.read_csv("constructors.csv", low_memory=False)

results_df = pd.read_csv("results.csv", low_memory=False)
races_df = pd.read_csv("races.csv", low_memory=False)
status_df = pd.read_csv("status.csv", low_memory=False)


drivers_raw_df["driver_name"] = (
    drivers_raw_df["forename"].fillna("").astype(str).str.strip() + " " +
    drivers_raw_df["surname"].fillna("").astype(str).str.strip()
).str.strip()

drivers_raw_df["driverRef"] = (
    drivers_raw_df["driverRef"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

drivers_df = drivers_raw_df[
    ["driverId", "driver_name"]
].drop_duplicates()

drivers_dropdown_df = drivers_raw_df[
    ["driverId", "driverRef", "driver_name"]
].drop_duplicates()


constructors_raw_df["constructorRef"] = (
    constructors_raw_df["constructorRef"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

constructors_raw_df["constructor_name"] = (
    constructors_raw_df["name"]
    .fillna(constructors_raw_df["constructorRef"])
    .astype(str)
    .str.strip()
)

constructors_df = constructors_raw_df[
    ["constructorId", "constructorRef", "constructor_name"]
].drop_duplicates()

constructors_df["constructorId"] = pd.to_numeric(
    constructors_df["constructorId"],
    errors="coerce"
)

constructors_df = constructors_df.dropna(subset=["constructorId"]).copy()
constructors_df["constructorId"] = constructors_df["constructorId"].astype(int)


print("Feature engineered dataset columns:")
print(data.columns.tolist())

print("Drivers dataset columns:")
print(drivers_df.columns.tolist())

print("Constructors dataset columns:")
print(constructors_df.columns.tolist())

def preprocess_for_ranking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns].copy()

    for col in df.columns:
        if col in label_encoders:
            mapping = label_encoders[col]
            fallback_value = 0

            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: mapping.get(x, fallback_value))
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

def get_constructor_name(row) -> str:
    if "constructor_name" in row and pd.notna(row["constructor_name"]):
        name = str(row["constructor_name"]).strip()
        if name != "" and name.lower() != "nan":
            return name

    if "constructor" in row and pd.notna(row["constructor"]):
        name = str(row["constructor"]).strip()
        if name != "" and name.lower() != "nan":
            return name

    if "constructorRef" in row and pd.notna(row["constructorRef"]):
        name = str(row["constructorRef"]).strip()
        if name != "" and name.lower() != "nan":
            return name

    return "Unknown Constructor"


def get_constructor_id(row) -> int:
    if "constructorId" in row and pd.notna(row["constructorId"]):
        try:
            return int(row["constructorId"])
        except Exception:
            return 0

    return 0


def add_constructor_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "constructorId" not in df.columns:
        df["constructorId"] = 0

    df["constructorId"] = pd.to_numeric(
        df["constructorId"],
        errors="coerce"
    ).fillna(0).astype(int)

    cols_to_drop = []

    for col in ["constructor_name", "constructorRef", "constructor"]:
        if col in df.columns:
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = df.merge(
        constructors_df,
        on="constructorId",
        how="left"
    )

    df["constructor_name"] = (
        df["constructor_name"]
        .fillna("Unknown Constructor")
        .astype(str)
        .str.strip()
    )

    df["constructorRef"] = (
        df["constructorRef"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    df["constructor"] = df["constructor_name"]

    return df

comparison_df = results_df.copy()

comparison_df = comparison_df.merge(
    drivers_raw_df[["driverId", "driverRef", "driver_name"]],
    on="driverId",
    how="left"
)

comparison_df = comparison_df.merge(
    constructors_df,
    on="constructorId",
    how="left"
)

comparison_df = comparison_df.merge(
    races_df[["raceId", "year", "round", "name"]],
    on="raceId",
    how="left"
)

comparison_df = comparison_df.rename(columns={"name": "race_name"})

comparison_df["points"] = pd.to_numeric(
    comparison_df["points"],
    errors="coerce"
).fillna(0)

comparison_df["positionOrder"] = pd.to_numeric(
    comparison_df["positionOrder"],
    errors="coerce"
)

comparison_df["grid"] = pd.to_numeric(
    comparison_df["grid"],
    errors="coerce"
).fillna(0)

comparison_df["year"] = pd.to_numeric(
    comparison_df["year"],
    errors="coerce"
)

comparison_df["round"] = pd.to_numeric(
    comparison_df["round"],
    errors="coerce"
)

comparison_df["driver_name"] = comparison_df["driver_name"].fillna("Unknown Driver")
comparison_df["driverRef"] = comparison_df["driverRef"].fillna("Unknown")
comparison_df["constructor_name"] = comparison_df["constructor_name"].fillna("Unknown Constructor")
comparison_df["constructorRef"] = comparison_df["constructorRef"].fillna("Unknown")
comparison_df["race_name"] = comparison_df["race_name"].fillna("Unknown Race")

def get_driver_summary(driver_id: int):
    driver_data = comparison_df[comparison_df["driverId"] == driver_id].copy()

    if driver_data.empty:
        return None

    driver_name = str(driver_data["driver_name"].iloc[0])
    driver_ref = str(driver_data["driverRef"].iloc[0])

    total_races = driver_data["raceId"].nunique()
    total_points = float(driver_data["points"].sum())

    wins = int((driver_data["positionOrder"] == 1).sum())
    podiums = int((driver_data["positionOrder"] <= 3).sum())

    valid_finishes = driver_data.dropna(subset=["positionOrder"]).copy()

    if len(valid_finishes) > 0:
        average_finish = round(float(valid_finishes["positionOrder"].mean()), 2)
        best_finish = int(valid_finishes["positionOrder"].min())
        worst_finish = int(valid_finishes["positionOrder"].max())
    else:
        average_finish = None
        best_finish = None
        worst_finish = None

    constructors = (
        driver_data["constructor_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    season_points = (
        driver_data.groupby("year", as_index=False)["points"]
        .sum()
        .sort_values("year")
    )

    season_points_list = []

    for _, row in season_points.iterrows():
        if pd.notna(row["year"]):
            season_points_list.append({
                "year": int(row["year"]),
                "points": float(row["points"])
            })

    season_finishes = (
        driver_data.dropna(subset=["positionOrder", "year"])
        .groupby("year", as_index=False)["positionOrder"]
        .mean()
        .sort_values("year")
    )

    average_finish_by_season = []

    for _, row in season_finishes.iterrows():
        if pd.notna(row["year"]):
            average_finish_by_season.append({
                "year": int(row["year"]),
                "averageFinish": round(float(row["positionOrder"]), 2)
            })

    return {
        "driverId": int(driver_id),
        "driverName": driver_name,
        "driverRef": driver_ref,
        "totalRaces": int(total_races),
        "totalPoints": round(total_points, 2),
        "wins": wins,
        "podiums": podiums,
        "averageFinish": average_finish,
        "bestFinish": best_finish,
        "worstFinish": worst_finish,
        "constructors": constructors,
        "seasonPoints": season_points_list,
        "averageFinishBySeason": average_finish_by_season
    }


@app.get("/")
def home():
    return {"message": "FastAPI is running"}

@app.get("/predict-league-table")
def predict_league_table():
    try:
        df = data.copy()

        required_cols = ["year", "round", "driverId"]
        missing_required = [col for col in required_cols if col not in df.columns]

        if missing_required:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Dataset missing required columns",
                    "missing_columns": missing_required,
                    "available_columns": df.columns.tolist()
                }
            )

        df = df.sort_values(["year", "round"])
        df = df.groupby("driverId", as_index=False).tail(1).copy()

        df = df.merge(drivers_df, on="driverId", how="left")

        df = add_constructor_names(df)

        display_df = df.copy()
        model_df = preprocess_for_ranking(df)

        display_df["ranking_score"] = ranker.predict(model_df)

        display_df = display_df.sort_values(
            "ranking_score",
            ascending=False
        ).reset_index(drop=True)

        display_df = display_df.head(20).copy()
        display_df["rank"] = display_df.index + 1

        results = []

        for _, row in display_df.iterrows():
            constructor_name = get_constructor_name(row)

            results.append({
                "rank": int(row["rank"]),
                "driverId": int(row["driverId"]) if pd.notna(row["driverId"]) else 0,
                "driver": str(row["driver_name"]) if pd.notna(row["driver_name"]) else str(row["driverId"]),
                "constructorId": get_constructor_id(row),
                "constructor": constructor_name,
                "constructor_name": constructor_name,
                "ranking_score": round(float(row["ranking_score"]), 4)
            })

        return {
            "status": "success",
            "count": len(results),
            "data": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/most-improved-drivers")
def most_improved_drivers():
    try:
        # Use raw race results because this always has driverId and constructorId
        df = results_df.copy()

        # Merge driver names
        df = df.merge(
            drivers_raw_df[["driverId", "driverRef", "driver_name"]],
            on="driverId",
            how="left"
        )

        # Merge constructor names
        df = df.merge(
            constructors_df[["constructorId", "constructorRef", "constructor_name"]],
            on="constructorId",
            how="left"
        )

        required_cols = [
            "driverId",
            "driver_name",
            "constructorId",
            "constructor_name",
            "grid",
            "positionOrder"
        ]

        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required columns",
                    "missing_columns": missing,
                    "available_columns": df.columns.tolist()
                }
            )

        df["driverId"] = pd.to_numeric(df["driverId"], errors="coerce")
        df["constructorId"] = pd.to_numeric(df["constructorId"], errors="coerce")
        df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
        df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")

        df = df.dropna(
            subset=[
                "driverId",
                "constructorId",
                "grid",
                "positionOrder"
            ]
        ).copy()

        # Only valid race starts and valid finishing positions
        df = df[(df["grid"] > 0) & (df["positionOrder"] > 0)].copy()

        df["driverId"] = df["driverId"].astype(int)
        df["constructorId"] = df["constructorId"].astype(int)

        df["driver_name"] = (
            df["driver_name"]
            .fillna(df["driverRef"])
            .fillna("Unknown Driver")
            .astype(str)
            .str.strip()
        )

        df["constructor_name"] = (
            df["constructor_name"]
            .fillna(df["constructorRef"])
            .fillna("Unknown Constructor")
            .astype(str)
            .str.strip()
        )

        df["constructorRef"] = (
            df["constructorRef"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )


        df["positions_gained"] = df["grid"] - df["positionOrder"]

        if "resultId" in df.columns:
            df = df.sort_values("resultId")
        else:
            df = df.sort_values("driverId")

        improved_drivers = (
            df.groupby(["driverId", "driver_name"], as_index=False)
            .agg(
                avg_positions_gained=("positions_gained", "mean"),
                total_positions_gained=("positions_gained", "sum"),
                races=("driverId", "count"),
                constructorId=("constructorId", "last"),
                constructor_name=("constructor_name", "last"),
                constructorRef=("constructorRef", "last")
            )
        )

        improved_drivers = improved_drivers[
            improved_drivers["races"] >= 5
        ].copy()

        improved_drivers = improved_drivers.sort_values(
            by=["avg_positions_gained", "total_positions_gained"],
            ascending=False
        )

        top10 = improved_drivers.head(10).copy()

        top10["avg_positions_gained"] = top10["avg_positions_gained"].round(2)
        top10["total_positions_gained"] = top10["total_positions_gained"].round(2)

        result = []

        for _, row in top10.iterrows():
            constructor_name = str(row["constructor_name"]).strip()

            if constructor_name == "" or constructor_name.lower() == "nan":
                constructor_name = "Unknown Constructor"

            result.append({
                "driverId": int(row["driverId"]),
                "driverName": str(row["driver_name"]),
                "constructorId": int(row["constructorId"]),
                "constructor": constructor_name,
                "constructor_name": constructor_name,
                "constructorRef": str(row["constructorRef"]),
                "avg_positions_gained": float(row["avg_positions_gained"]),
                "total_positions_gained": float(row["total_positions_gained"]),
                "races": int(row["races"])
            })

        return {
            "status": "success",
            "count": len(result),
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drivers")
def get_drivers():
    try:
        df = drivers_dropdown_df.copy()

        df = df.rename(columns={
            "driver_name": "driverName"
        })

        df["driverId"] = pd.to_numeric(df["driverId"], errors="coerce")
        df = df.dropna(subset=["driverId"]).copy()
        df["driverId"] = df["driverId"].astype(int)

        df["driverName"] = df["driverName"].fillna("Unknown Driver").astype(str)
        df["driverRef"] = df["driverRef"].fillna("Unknown").astype(str)

        df = df.sort_values("driverName")

        result = []

        for _, row in df.iterrows():
            result.append({
                "driverId": int(row["driverId"]),
                "driverRef": str(row["driverRef"]),
                "driverName": str(row["driverName"])
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare-drivers")
def compare_drivers(driver1_id: int, driver2_id: int):
    try:
        if driver1_id == driver2_id:
            raise HTTPException(
                status_code=400,
                detail="Please select two different drivers."
            )

        driver1 = get_driver_summary(driver1_id)
        driver2 = get_driver_summary(driver2_id)

        if driver1 is None:
            raise HTTPException(
                status_code=404,
                detail=f"Driver with ID {driver1_id} not found."
            )

        if driver2 is None:
            raise HTTPException(
                status_code=404,
                detail=f"Driver with ID {driver2_id} not found."
            )

        return {
            "status": "success",
            "driver1": driver1,
            "driver2": driver2
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/race-results-history")
def get_race_results_history(
    year: int | None = None,
    driver: str | None = None,
    constructor: str | None = None,
    limit: int = 100
):
    try:
        df = comparison_df.copy()

        df = df.merge(
            status_df[["statusId", "status"]],
            on="statusId",
            how="left"
        )

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
        df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
        df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)

        df["final_position"] = df["positionOrder"]

        df["race_name"] = df["race_name"].fillna("Unknown Race").astype(str)
        df["driver_name"] = df["driver_name"].fillna("Unknown Driver").astype(str)
        df["driverRef"] = df["driverRef"].fillna("Unknown").astype(str)
        df["constructor_name"] = df["constructor_name"].fillna("Unknown Constructor").astype(str)
        df["constructorRef"] = df["constructorRef"].fillna("Unknown").astype(str)
        df["status"] = df["status"].fillna("Unknown").astype(str)

        if year is not None:
            df = df[df["year"] == year]

        if driver is not None and driver.strip() != "":
            driver_search = driver.strip()

            df = df[
                df["driver_name"].str.contains(driver_search, case=False, na=False)
                | df["driverRef"].str.contains(driver_search, case=False, na=False)
            ]

        if constructor is not None and constructor.strip() != "":
            constructor_search = constructor.strip()

            df = df[
                df["constructor_name"].str.contains(constructor_search, case=False, na=False)
                | df["constructorRef"].str.contains(constructor_search, case=False, na=False)
            ]

        df = df.sort_values(
            by=["year", "round", "final_position"],
            ascending=[False, True, True]
        )

        df = df.head(limit)

        response = []

        for _, row in df.iterrows():
            response.append({
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "round": int(row["round"]) if pd.notna(row["round"]) else None,
                "race_name": str(row["race_name"]),

                "driverId": int(row["driverId"]) if pd.notna(row["driverId"]) else 0,
                "driver_name": str(row["driver_name"]),
                "driverRef": str(row["driverRef"]),

                "constructorId": int(row["constructorId"]) if pd.notna(row["constructorId"]) else 0,
                "constructor": str(row["constructor_name"]),
                "constructor_name": str(row["constructor_name"]),
                "constructorRef": str(row["constructorRef"]),

                "grid": int(row["grid"]) if pd.notna(row["grid"]) else None,
                "final_position": int(row["final_position"]) if pd.notna(row["final_position"]) else None,
                "points": float(row["points"]) if pd.notna(row["points"]) else 0.0,
                "status": str(row["status"])
            })

        return {
            "status": "success",
            "count": len(response),
            "results": response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))