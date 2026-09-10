from flask import Flask, render_template, request,redirect,session
import joblib
import os

app = Flask(__name__)
app.secret_key="iris-secret-key"

# Load the trained Iris model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_model.pkl")
model = joblib.load(MODEL_PATH)

# Iris class names (used if the model returns 0, 1, or 2)
iris_names = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


@app.route("/")
def home():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    if username == "admin" and password == "iris123":
        session["logged_in"] = True
        return redirect("/home")

    return render_template("login.html", error="Invalid username or password")


@app.route("/home")
def dashboard():
    if not session.get("logged_in"):
        return redirect("/") 
    return render_template("index.html")       


@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # The order must match the order used while training the model
        features = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        prediction = model.predict(features)[0]

        # Convert numeric prediction to Iris name
        if isinstance(prediction, (int, float)) and int(prediction) in iris_names:
            result = iris_names[int(prediction)]
        else:
            result = str(prediction)

        return render_template(
            "index.html",
            prediction=result,
            values={
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numbers in all four fields."
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)