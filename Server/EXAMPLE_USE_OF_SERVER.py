import requests

# Example of how to use the ranking endpoint with Ray Sweeping method
# data = {
#     "constraints": [[1, 2, "<="], [1, 1, ">="]],
#     "method": "Ray Sweeping",
#     "columns": ["rapid_win", "bullet_win"],
#     "num_ret_tuples": 1,
#     "num_of_rankings": 1
# }
# response = requests.post("http://127.0.0.1:8000/ranking", json=data)




# Example of how to use the ranking endpoint with Randomized Rounding method
# data = {
#     "constraints": [[1, 2, "<="], [1, 1, ">="]],
#     "method": "Randomized Rounding",
#     "columns": ["rapid_win", "bullet_win"],
#     "num_ret_tuples": 3,
#     "num_of_rankings": 2,
#     "num_of_samples": 1000
# }
# response = requests.post("http://127.0.0.1:8000/ranking", json=data)

# Example of how to use the stability endpoint
# data = {
#     "columns": ["rapid_win", "bullet_win"],
#     "W1": 0.2,  # Replace with an appropriate integer value
#     "W2": 0.8   # Replace with an appropriate integer value
# }
# response = requests.post("http://127.0.0.1:8000/stability", json=data)


# Example of how to use the columns endpoint
response = requests.get("http://127.0.0.1:8000/columns")

# Print the JSON response
print(response.json())

