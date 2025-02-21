### implementation of the backend services from the paper.

import math
import numpy as np
import pandas as pd
import json
import heapq
from fastapi import FastAPI
import os
import hashlib


cleaned_file_path = os.path.join(os.path.dirname(__file__), "Data", "GM_players_statistics_cleaned.csv")
cleaned_df = pd.read_csv(cleaned_file_path)

original_file_path = os.path.join(os.path.dirname(__file__), "Data", "GM_players_statistics.csv")
original_df = pd.read_csv(original_file_path)


def generate_ranking_id(ranking):
    """Generates a unique ID for a ranking by hashing the tuple."""

    ranking_str = ",".join(map(str, ranking))  # Convert to string
    return hashlib.md5(ranking_str.encode()).hexdigest()  # Hash it

def validate_input(columns, cleaned_df):
    """ Validates input columns to ensure they exist in the dataset and are exactly two. """
    if len(columns) != 2 or not all(col in cleaned_df.columns for col in columns):
        raise ValueError("Columns not found in the data or invalid number of columns provided")

def compute_feasible_region(constraints):
    """ Computes the feasible angle region based on constraints. Raises an error if infeasible. """
    if constraints:
        region_in_angles = find_feasible_angle_region(constraints)
        if region_in_angles is None:
            raise ValueError("No feasible region found - Infeasible constraints")
        return region_in_angles
    return None

def process_ray_sweeping(region_in_angles, columns, num_of_rankings, num_ret_tuples):
    """ Processes the ranking using the Ray Sweeping method and returns results. """
    ranking_heap = ray_sweeping(region_in_angles, columns)
    results = []

    for _ in range(num_of_rankings):
        stability, angle_range_rank = heapq.heappop(ranking_heap)
        stability = -stability
        angle = np.mean(angle_range_rank)
        ranking_function = from_angle_to_vector(angle)
        ranking = find_ranking(ranking_function, columns).reset_index(drop=True)

        final_df = ranking.loc[:num_ret_tuples-1, ["user_id", "name", "Rank", columns[0], columns[1]]]
        results.append(format_ranking_output(final_df, ranking_function, stability))

    return results

def process_randomized_rounding(region_in_angles, columns, num_of_rankings, num_ret_tuples, num_of_samples, k_sample):
    """ Processes the ranking using the Randomized Rounding method and returns results. """
    if k_sample is None or k_sample > 100:
        k_sample = 100

    ranking_list = randomized_get_next(region_in_angles, columns, num_of_rankings, num_ret_tuples, num_of_samples, k_sample)
    results = []

    for rank in ranking_list:
        angle, stability = rank
        ranking_function = from_angle_to_vector(angle)
        ranking = find_ranking(ranking_function, columns).reset_index(drop=True)

        final_df = ranking.loc[:num_ret_tuples-1, ["user_id", "name", "Rank", columns[0], columns[1]]]
        results.append(format_ranking_output(final_df, ranking_function, stability))

    return results

def format_ranking_output(final_df, ranking_function, stability):
    """ Formats ranking output into dictionary format. """
    return {
        "Ranking": final_df.to_dict(orient="records"),
        "Ranking_Function": {"w1": ranking_function[0], "w2": ranking_function[1]},
        "Stability": stability
    }

def save_results_to_json(results, filename="res.json"):
    """ Saves results to a JSON file. """
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

def sort_data(constraints: list, method: str, columns: list, num_ret_tuples: int, num_of_rankings : int, num_of_samples : int = None, k_sample=None) -> tuple[pd.DataFrame, float]:
    """
    Sorts the data based on the constraints and method provided.

    Parameters:
        constraints (list): List of constraints that define the feasible ranking region.
        method (str): Sorting method, either "Ray Sweeping" or "Randomized Rounding".
        columns (list): List containing exactly two column names to be used for ranking.
        num_ret_tuples (int): Number of top-ranked tuples to return.
        num_of_rankings (int): Number of top stable rankings to compute.
        num_of_samples (int, optional): Number of samples used in randomized rounding (only applicable for "Randomized Rounding").
        k_sample (int, optional): Maximum sample size for randomized ranking.

    Returns:
        tuple[pd.DataFrame, float]: A tuple containing:
            - The ranked DataFrame.
            - The stability score of the ranking.
    """

    validate_input(columns, cleaned_df)
    region_in_angles = compute_feasible_region(constraints)

    # Handle sorting based on the selected method
    if method == "Ray Sweeping":
        results = process_ray_sweeping(region_in_angles, columns, num_of_rankings, num_ret_tuples)
    elif method == "Randomized Rounding":
        results = process_randomized_rounding(region_in_angles, columns, num_of_rankings, num_ret_tuples, num_of_samples, k_sample)
    else:
        raise ValueError(f"Invalid method provided - {method}, choose from 'Ray_Sweeping' or 'Randomized_Rounding'")
    
    save_results_to_json(results)
    return results

def get_ranking_stability(W1, W2, columns):

    angle = find_angle(W1, W2)

    if angle < 0 or angle > 90:
        raise ValueError("Invalid angle provided")

    ranking = find_ranking(from_angle_to_vector(angle), columns)
    print(f"We have a ranking with length {len(ranking)} and the angle is {angle}")

    max_angle = 90
    min_angle = 0

    for i in range(len(ranking)-1):
        
        exch_angle = calculate_exchange_ordering_angle([min_angle, max_angle], [i, i+1], columns)
        if exch_angle is not None:
            print(f"Angle between {i} and {i+1} is {exch_angle}")
            if exch_angle < angle and exch_angle > min_angle:
                min_angle = exch_angle
            elif exch_angle > angle and exch_angle < max_angle:
                max_angle = exch_angle

    print(f"Mimimum angle is {min_angle} and maximum angle is {max_angle}")

    res = (max_angle - min_angle)/ 90
    return {'stability': res}

def calculate_exchange_ordering_angle(region_in_angles, indexes, columns):
    """
    Calculates the exchange ordering angle for a pair of players.
    """
    item_1 = cleaned_df.loc[indexes[0], columns]
    item_2 = cleaned_df.loc[indexes[1], columns]

    # check dominance
    if item_1[columns[0]] >= item_2[columns[0]] and item_1[columns[1]] >= item_2[columns[1]]:
        # print(f"player {indexes[0]} dominates player {indexes[1]}")
        # print(f"values: {float(item_1[columns[0]]), float(item_1[columns[1]])} >= {float(item_2[columns[0]]), float(item_2[columns[1]])}")
        return None
    
    angle = compute_first_quadrant_angle(item_1, item_2, columns)

    # check if angle is within the feasible region
    if (region_in_angles[0] <= angle) and (angle <= region_in_angles[1]):
        # print(f"Angle between {indexes[0]} and {indexes[1]} is {angle}")
        return angle
    
    else:
        # print(f"Angle between {indexes[0]} and {indexes[1]} is {angle} outside the feasible region")
        # print(f"values: {float(item_1[columns[0]]), float(item_1[columns[1]])} and {float(item_2[columns[0]]), float(item_2[columns[1]])}")
        return None

def compute_first_quadrant_angle(item_1, item_2, columns):
    delta_x = item_2[columns[0]] - item_1[columns[0]]
    delta_y = item_2[columns[1]] - item_1[columns[1]]
    
    angle = np.degrees(np.arctan2(delta_y, delta_x))  # Compute angle in degrees
    
    # Convert to first quadrant representation
    if angle < 0:
        angle += 360  # Convert negative angles to positive
    
    if angle > 90:
        angle = 180 - angle if angle <= 180 else angle - 180
        angle = 90 - angle if angle > 90 else angle  # Keep it within 0-90 range
    
    return angle
    
def ray_sweeping(region_in_angles : list, columns : str) -> pd.DataFrame:

    # print(f"range is between {region_in_angles[0]} and {region_in_angles[1]}")

    init_rank = find_ranking(from_angle_to_vector(region_in_angles[0]), columns)
    min_heap = []

    for i in range(len(init_rank)-1):
        angle = calculate_exchange_ordering_angle(region_in_angles, [i, i+1], columns)
        if angle is not None:
            heapq.heappush(min_heap, (angle, [i, i+1]))

    old_angle = float(region_in_angles[0])  # Convert to Python float
    max_heap = []
    range_area = float(region_in_angles[1]) - old_angle  # Ensure subtraction is between Python floats

    while len(min_heap) > 0:
        angle, indexes = heapq.heappop(min_heap)
        index_1, index_2 = indexes
        stability = (float(angle) - old_angle) / range_area  # Convert angle to float
        heapq.heappush(max_heap, (-stability, [old_angle, float(angle)]))  # Convert angle to float
        old_angle = float(angle)  # Ensure old_angle remains a Python float

    # while len(max_heap) > 0:
    #     stability, angle_range_rank = heapq.heappop(max_heap)
    #     print(f"Stability: {-stability} for range {angle_range_rank}")
    
    return max_heap

def generate_randomized_angles(region_in_angles : list, num_of_smaples : int) -> list:
    res  = []
    for i in range(num_of_smaples):
        angle = np.random.uniform(region_in_angles[0], region_in_angles[1])
        res.append(angle)
    
    return res

def randomized_get_next(region_in_angles : list, columns : str, num_of_rankings : int, num_ret_tuples : int, num_of_smaples : int, k_sample : int) -> list:
    
    samples = generate_randomized_angles(region_in_angles, num_of_smaples)
    ranking_counts = {}
    ranking_angles = {}

    for sample in samples:
        ranking = find_ranking(from_angle_to_vector(sample), columns)
        ranking = ranking.iloc[:k_sample]['user_id']
        ranking_id = generate_ranking_id(ranking)

        ranking_angles[ranking_id] = sample

        # Update ranking occurrence count
        if ranking_id in ranking_counts:
            ranking_counts[ranking_id] += 1
        else:
            ranking_counts[ranking_id] = 1

    # # print the ranking counts
    # for ranking_id, count in ranking_counts.items():
    #     print(f"Ranking {ranking_id} occurs {count} times")

    # # print the ranking angles
    # for ranking_id, angle in ranking_angles.items():
    #     print(f"Ranking {ranking_id} has angle {angle}")

    ranking_list = []
    if num_of_rankings == None:
        num_of_rankings = len(ranking_counts)
    for i in range(num_of_rankings):
        try:
            max_ranking = max(ranking_counts, key=ranking_counts.get)
        except ValueError:
            break
        angle = ranking_angles[max_ranking]
        stability = ranking_counts[max_ranking] / num_of_smaples
        rank = [angle, stability]
        ranking_list.append(rank)
        del ranking_counts[max_ranking]

    #print the ranking list
    # for rank in ranking_list:
    #     print(f"Ranking {rank[0]} has stability {rank[1]}")

    return ranking_list
        
def find_ranking(waights: list, columns: list) -> pd.DataFrame:
    if len(waights) != 2:
        raise ValueError("Invalid number of weights provided")
    
    if not math.isclose(waights[0] + waights[1], 1, rel_tol=1e-6):
        raise ValueError("Weights should sum to approximately 1")
    
    df_ranked = cleaned_df.copy()
    df_ranked["Rank"] = df_ranked[columns[0]]*waights[0] + df_ranked[columns[1]]*waights[1]
    df_ranked = df_ranked.sort_values(by="Rank", ascending=False)
    return df_ranked

# def find_angle(a, b):
#     return np.degrees(np.arctan2(b, a))

def find_angle(W1, W2):
    print(f"DEBUG: Inside find_angle, W1={W1}, W2={W2}, Type(W1)={type(W1)}, Type(W2)={type(W2)}")

    assert isinstance(W1, (int, float)), f"Error: W1 is {type(W1)}, expected int/float"
    assert isinstance(W2, (int, float)), f"Error: W2 is {type(W2)}, expected int/float"

    W1 = float(W1)  # Explicit conversion
    W2 = float(W2)

    return np.degrees(np.arctan2(W2, W1))

def find_feasible_angle_region(constraints):
    """
    Finds the intersection of angle constraints given as (a, b, sign).
    """

    theta_min = 0
    theta_max = 90

    for a, b, sign in constraints:
        angle = find_angle(a, b)
        if sign == "<=":
            theta_max = min(theta_max, angle)
        elif sign == ">=":
            theta_min = max(theta_min, angle)

    if theta_min >= theta_max:
        return None

    return theta_min, theta_max

def from_angle_to_vector(angle):
    w_1 = math.cos(math.radians(angle))
    w_2 = math.sin(math.radians(angle))
    
    # Normalize to make sure w_1 + w_2 = 1
    total = w_1 + w_2
    return [w_1 / total, w_2 / total]

def get_columns_names():
    return cleaned_df.columns.tolist()

def sample_first_five_entries():
    """
    Returns the first five entries from the dataset.
    """
    
    return original_df.head(5).replace({np.nan: None}).to_dict(orient="records")

# def main():

    # res = sort_data([(1,2,"<="),(1, 1, ">=")], "Randomized Rounding", ["followers", "bullet_win"], num_of_rankings=2, num_ret_tuples=3, num_of_smaples=9000, k_sample=40)
    # print(res)

    # print(f"Stability score: {res[1]}")
    # stability = get_ranking_stability(0.25, 0.75, ["followers", "bullet_win"])
    # print(f"Stability score: {stability}")


# if __name__ == "__main__":
    # main()
