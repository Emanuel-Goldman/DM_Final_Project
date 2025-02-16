import requests

# # Define the JSON payload
# data = {
#     "constraints": [[1, 2, "<="], [1, 1, ">="]],
#     "method": "Ray Sweeping",
#     "columns": ["rapid_win", "bullet_win"],
#     "num_ret_tuples": 1,
#     "num_of_rankings": 1
# }

# # Send POST request to the ranking endpoint
# response = requests.post("http://127.0.0.1:8000/ranking", json=data)




# Define the JSON payload
data = {
    "constraints": [[1, 2, "<="], [1, 1, ">="]],
    "method": "Randomized Rounding",
    "columns": ["rapid_win", "bullet_win"],
    "num_ret_tuples": 3,
    "num_of_rankings": 2,
    "num_of_samples": 1000
}

# Send POST request to the ranking endpoint
response = requests.post("http://127.0.0.1:8000/ranking", json=data)

# Print the JSON response
print(response.json())

