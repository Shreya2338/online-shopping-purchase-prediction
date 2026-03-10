import pandas as pd
from lightgbm import LGBMClassifier


# LOAD DATA!!


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# TARGET


y = train["purchased"]


# DROP NON-FEATURES


X = train.drop(columns=["purchased", "session_id"])
X_test = test.drop(columns=["session_id"])


# FEATURE ENGINEERING


X["total_pages"] = (
    X["administrative"] +
    X["informational"] +
    X["product_related"]
)

X["total_duration"] = (
    X["administrative_duration"] +
    X["informational_duration"] +
    X["product_related_duration"]
)

X_test["total_pages"] = (
    X_test["administrative"] +
    X_test["informational"] +
    X_test["product_related"]
)

X_test["total_duration"] = (
    X_test["administrative_duration"] +
    X_test["informational_duration"] +
    X_test["product_related_duration"]
)


# ONE HOT ENCODING

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)


# MODEL


model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=-1,
    class_weight="balanced",
    random_state=42
)

model.fit(X, y)


# PREDICT


preds = model.predict_proba(X_test)[:,1]


# SUBMISSION FILE


submission = pd.DataFrame({
    "session_id": test["session_id"],
    "purchase_probability": preds
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")
