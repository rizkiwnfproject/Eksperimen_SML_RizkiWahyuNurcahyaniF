import requests

data = {
    "inputs": [
        [
            0.0,  # Age
            1,  # Sex
            2,  # ChestPainType
            1.0,  # RestingBP
            -0.87,  # Cholesterol
            0,  # FastingBS
            1,  # RestingECG
            -0.45,  # MaxHR
            0,  # ExerciseAngina
            -0.33,  # Oldpeak
            2,  # ST_Slope
        ]
    ]
}

res = requests.post("http://localhost:8000/predict", json=data)
print(res.json())
