### implementation of the backend services from the paper.

import math
from typing import List, Literal, NamedTuple, Tuple, TypedDict
import numpy as np
import pandas as pd
import json
import heapq
from fastapi import FastAPI
import os
import hashlib

# ------------------------- Custom Type Definitions -------------------------

class Constraint: [int, int, str] # [a1, a2, op]

# ------------------------- DataFrame Init -------------------------

cleaned_file_path = os.path.join(os.path.dirname(__file__), "Data", "GM_players_statistics_cleaned.csv")
cleaned_df = pd.read_csv(cleaned_file_path)

original_file_path = os.path.join(os.path.dirname(__file__), "Data", "GM_players_statistics.csv")
original_df = pd.read_csv(original_file_path)

# ------------------------- Backend Services Implementation -------------------------

def generate_ranking_id(ranking):
    """Generates a unique ID for a ranking by hashing the tuple."""

    ranking_str = ",".join(map(str, ranking))  # Convert to string
    return hashlib.md5(ranking_str.encode()).hexdigest()  # Hash it

def validate_input(columns: list[str], cleaned_df: pd.DataFrame):
    """ Validates input columns to ensure they exist in the dataset and are exactly two. """

    if len(columns) != 2 or not all(col in cleaned_df.columns for col in columns):
        raise ValueError("Columns not found in the data or invalid number of columns provided")

def find_angle(a1: float, a2: float) -> float:
    """Returns the angle created by the line 'a1*w1=a2*w2' and the -axis"""
    
    return np.degrees(np.arctan2(a2, a1))

    # If the feasible region is empty (min >= max), return None
    return (theta_min, theta_max) if theta_min < theta_max else None

def from_angle_to_vector(angle: int) -> Tuple[float, float]:
    """
    Converts an angle in degrees to a 2D vector on the unit circle.

    Returns a tuple containing the normalized vector components (w1, w2),
    where w1 and w2 represent the cosine and sine of the angle, respectively.
    """

    w1 = math.cos(math.radians(angle))
    w2 = math.sin(math.radians(angle))
    
    # Normalize to make sure w1 + w2 = 1
    total = w1 + w2
    return [w1 / total, w2 / total]

def compute_feasible_region(constraints: List[Constraint]) -> (Tuple[int, int] | None):
    """
    Computes and returns a tuple (theta_min, theta_max) representing the valid range of angles 
    based on constraints of the form: `a1 * w1 op a2 * w2`, 
    where `op` is a comparison operator (e.g., `<=`, `>=`, `<`, `>`).
    
    Raises a ValueError if no feasible region is found due to infeasible constraints.
    """

    # Initialize the feasible region to the full range of angles (0° to 90°) in the first quadrant
    theta_min = 0
    theta_max = 90

    for a1, a2, op in constraints:
        # Compute the angle of the line defined by a1 * w1 = a2 * w2
        angle = find_angle(a1, a2)

        # Update the feasible region based on the constraint operator
        if op in ("<=", "<"): 
            theta_max = min(theta_max, angle)
        elif op in (">=", ">"):
            theta_min = max(theta_min, angle)

    # Check if the feasible region is empty (min >= max) and raise an error if so
    if theta_min >= theta_max:
        raise ValueError("No feasible region found - Infeasible constraints")

    # Return the valid region of angles
    return (theta_min, theta_max)

def process_raysweeping(region_in_angles, columns, num_of_rankings, num_ret_tuples):
    """Processes the ranking using the Ray Sweeping method and returns results."""

    ranking_heap = compute_raysweeping_heap(region_in_angles, columns)
    results = []

    for _ in range(num_of_rankings):
        stability, angle_range_rank = heapq.heappop(ranking_heap)
        stability = -stability
        angle = np.mean(angle_range_rank)
        ranking_function = from_angle_to_vector(angle)
        ranking = find_ranking(ranking_function, columns).reset_index(drop=True)

        final_df = ranking.loc[:num_ret_tuples-1, ["user_id", "name", "rank", columns[0], columns[1]]]
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

        final_df = ranking.loc[:num_ret_tuples-1, ["user_id", "name", "rank", columns[0], columns[1]]]
        results.append(format_ranking_output(final_df, ranking_function, stability))

    return results

def format_ranking_output(final_df, ranking_function, stability):
    """ Formats ranking output into dictionary format. """

    return {
        "ranked_list": final_df.to_dict(orient="records"),
        "ranking_function": {"w1": ranking_function[0], "w2": ranking_function[1]},
        "stability": stability
    }

def save_results_to_json(results, filename="res.json"):
    """ Saves results to a JSON file. """

    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

def sort_data(constraints: List[Constraint], algorithm: str, columns: List[str], num_ret_tuples: int, num_of_rankings: int, num_of_samples: int = None, k_sample: int = None) -> tuple[pd.DataFrame, float]:
    """
    Sorts the data based on the constraints and method provided.

    Returns a tuple containing:
        - The ranked DataFrame.
        - The stability score of the ranking.
    """

    # Ensures input columns exist in the dataset and are exactly two
    validate_input(columns, cleaned_df)

    # Computes the feasible region of ranking functions
    region = compute_feasible_region(constraints)

    # Handle sorting based on the selected ranking algorithm
    if algorithm == "raysweeping":
        results = process_raysweeping(region, columns, num_of_rankings, num_ret_tuples)
    elif algorithm == "randomized-rounding":
        results = process_randomized_rounding(region, columns, num_of_rankings, num_ret_tuples, num_of_samples, k_sample)
    else:
        raise ValueError(f"Invalid method provided - {algorithm}, choose from 'raysweeping' or 'randomized-rounding'")
    
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
        
        exch_angle = calculate_ordering_exchange_angle([min_angle, max_angle], [i, i+1], columns)
        if exch_angle is not None:
            print(f"Angle between {i} and {i+1} is {exch_angle}")
            if exch_angle < angle and exch_angle > min_angle:
                min_angle = exch_angle
            elif exch_angle > angle and exch_angle < max_angle:
                max_angle = exch_angle

    print(f"Mimimum angle is {min_angle} and maximum angle is {max_angle}")

    res = (max_angle - min_angle)/ 90
    return {'stability': res}

def calculate_ordering_exchange_angle(region: Tuple[int,int], item_indices: Tuple[int,int], columns: List[str]):
    """
    Calculates the ordering exchange angle for a pair of players.

    Parameters:
        region (Tuple[int, int]): A tuple defining the feasible angle region (theta_min, theta_max).
        item_indices (Tuple[int, int]): A tuple containing the indices of the two items to compare.
        columns (List[str]): A list of the column names used to retrieve attributes of the items.

    Returns:
        float or None: The exchange angle if it is within the feasible region, otherwise None.
    """

    # Retrieve the attributes of the two items (players) based on their indices
    item1 = cleaned_df.loc[item_indices[0], columns]
    item2 = cleaned_df.loc[item_indices[1], columns]

    # Check for dominance: if item1 dominates item2 in both attributes, return None
    if item1[columns[0]] >= item2[columns[0]] and item1[columns[1]] >= item2[columns[1]]:
        return None
    
    angle = compute_first_quadrant_angle(item1, item2, columns)

    # Check if angle is within the feasible region
    if (region[0] <= angle) and (angle <= region[1]):
        return angle
    else:
        return None

def compute_first_quadrant_angle(item1, item2, columns: List[str]):
    """Computes the angle between two items based on their attributes in the first quadrant."""

    delta_x = item2[columns[0]] - item1[columns[0]]
    delta_y = item2[columns[1]] - item1[columns[1]]
    
    angle = np.degrees(np.arctan2(delta_y, delta_x))  # Compute angle in degrees
    
    # Convert to first quadrant representation
    if angle < 0:
        angle += 360  # Convert negative angles to positive
    
    if angle > 90:
        angle = 180 - angle if angle <= 180 else angle - 180
        angle = angle - 90 if angle > 90 else angle  # Keep it within 0-90 range
        
    return angle
    
def compute_raysweeping_heap(region: Tuple[int,int], columns: List[str]) -> pd.DataFrame:

    # Computes the initial order of the dataset items based on U[1] (min angle in given region)
    init_rank = find_ranking(from_angle_to_vector(region[0]), columns)

    min_heap = [] # Contains the ordering exchange angles for each pair of adjacent items in init_rank

    # Calculate ordering exchanges between every adjacent pair of items
    for i in range(len(init_rank)-1):
        angle = calculate_ordering_exchange_angle(region, [i, i+1], columns)
        if angle is not None:
            heapq.heappush(min_heap, (angle, [i, i+1]))

    old_angle = float(region[0])  # Convert to Python float
    max_heap = []
    range_area = float(region[1]) - old_angle  # Ensure subtraction is between Python floats

    while len(min_heap) > 0:
        angle, indexes = heapq.heappop(min_heap)
        index_1, index_2 = indexes
        stability = (float(angle) - old_angle) / range_area  # Convert angle to float
        heapq.heappush(max_heap, (-stability, [old_angle, float(angle)]))  # Convert angle to float
        old_angle = float(angle)  # Ensure old_angle remains a Python float
    
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

    return ranking_list
        
def find_ranking(weights: Tuple[float, float], columns: List[str]) -> pd.DataFrame:
    """
    Computes a ranking for entries in a DataFrame based on weighted scores of specified columns.
    Returns a DataFrame containing the original data with an additional 'rank' column, and sorted by rank.
    """

    if len(weights) != 2:
        raise ValueError("Invalid number of weights provided")
    
    if not math.isclose(weights[0] + weights[1], 1, rel_tol=1e-6):
        raise ValueError("Weights should sum to approximately 1")
    
    df_ranked = cleaned_df.copy()

    # Calculate the rank for each entry using the weighted sum of the specified columns
    df_ranked["rank"] = df_ranked[columns[0]]*weights[0] + df_ranked[columns[1]]*weights[1]

    # Sort the DataFrame by the 'rank' column in descending order
    df_ranked = df_ranked.sort_values(by="rank", ascending=False)
    return df_ranked

def get_columns_names() -> list[str]:
    """Returns a list of column names in the cleaned DataFrame"""

    return cleaned_df.columns.tolist()

def sample_first_five_entries():
    """Returns the first five entries from the dataset."""
    
    return original_df.head(5).replace({np.nan: None}).to_dict(orient="records")

