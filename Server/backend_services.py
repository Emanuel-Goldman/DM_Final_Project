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
from dataclasses import dataclass

# ------------------------- Custom Type Definitions -------------------------

class Constraint: [int, int, str] # [a1, a2, op]

@dataclass
class Region: 
    theta_min: float
    theta_max: float

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

def validate_input(columns: List[str], cleaned_df: pd.DataFrame):
    """ Validates input columns to ensure they exist in the dataset and are exactly two. """

    if len(columns) != 2 or not all(col in cleaned_df.columns for col in columns):
        raise ValueError("Columns not found in the data or invalid number of columns provided")

def find_angle(a1: float, a2: float) -> float:
    """Returns the angle created by the line 'a1*w1=a2*w2' and the w2-axis"""
    
    return float(np.degrees(np.arctan2(a1, a2)))

def from_angle_to_vector(angle: float) -> Tuple[float, float]:
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

def compute_interest_region(constraints: List[Constraint]) -> (Region | None):
    """
    Computes and returns a tuple (theta_min, theta_max) representing the valid range of angles 
    based on constraints of the form: `a1 * w1 op a2 * w2`, 
    where `op` is a comparison operator (e.g., `<=`, `>=`, `<`, `>`).
    
    Raises a ValueError if no interest region is found due to infeasible constraints.
    """

    # Initialize the interest region to the full range of angles (0° to 90°) in the first quadrant
    theta_min = 0
    theta_max = 90

    # for a1, a2, op in constraints:
    #     # Compute the angle of the line defined by a1 * w1 = a2 * w2
    #     angle = find_angle(a1, a2)

    #     print(f"The angle for constraint {a1}*w1={a2}*w2 is {angle}") # DEBUG

    #     # Update the feasible region based on the constraint operator
    #     if op in ("<=", "<", "=<"): 
    #         theta_max = min(theta_max, angle)
    #     elif op in (">=", ">", "=>"):
    #         theta_min = max(theta_min, angle)

    for constraint in constraints:
        print(f"Constraint: {constraint}")  # Print entire constraint object
        a1, a2, op = constraint
        print(f"Parsed values -> a1: {a1}, a2: {a2}, op: '{op}'")  

        angle = find_angle(a1, a2)
        print(f"find_angle({a1}, {a2}) = {angle}")

        if op in ("<=", "<", "=<"): 
            theta_min = max(theta_min, angle)
        elif op in (">=", ">", "=>"):
            theta_max = min(theta_max, angle)

    print(f"Updated region -> theta_min: {theta_min}, theta_max: {theta_max}")

    # Check if the feasible region is empty (min >= max) and raise an error if so
    if theta_min >= theta_max:
        raise ValueError("No feasible region found - Infeasible constraints")

    print(f"The interest region is {(theta_min, theta_max)}") # DEBUG

    # Return the valid region of angles
    return Region(theta_min, theta_max)

def process_raysweeping(interest_region: Region, columns: List[str], num_of_rankings: int, num_ret_tuples: int):
    """Processes the ranking using the Ray Sweeping method and returns results."""

    # Compute the ranking heap based on the given angle region and columns
    ranking_heap = compute_raysweeping_heap(interest_region, columns)
    results = []

    # Process the specified number of rankings
    for _ in range(num_of_rankings):
        # Extract the stability and angle range from the heap
        stability, angle_range_rank = heapq.heappop(ranking_heap)
        stability = -stability
        angle = np.mean([angle_range_rank.theta_min, angle_range_rank.theta_max]) # Compute the average angle of the range

        # Convert the average angle to a ranking function
        ranking_function = from_angle_to_vector(angle)

        # Find the ranking based on the calculated ranking function
        ranking = find_ranking(ranking_function, columns).reset_index(drop=True)

        # Select the top results based on the specified number of return tuples
        final_df = ranking.loc[:num_ret_tuples-1, ["user_id", "name", "rank", columns[0], columns[1]]]

        # Format the ranking output and add it to the results list
        results.append(format_ranking_output(final_df, ranking_function, stability, columns))

    return results

def process_randomized_rounding(interest_region: Region, columns: List[str], num_of_rankings: int, num_ret_tuples: int, num_of_samples: int, k_sample: int):
    """Processes the ranking using the Randomized Rounding method and returns results.
    
    This function generates rankings by sampling weight vectors from the given 
    interest region, computing rankings, and formatting the results.

    Args:
        interest_region (Region): The region defining the range of valid angles.
        columns (List[str]): The columns used for ranking computations.
        num_of_rankings (int): The number of rankings to generate.
        num_ret_tuples (int): The number of top-ranked players to return per ranking.
        num_of_samples (int): The number of weight function samples to generate.
        k_sample (int): The number of top players considered for ranking comparisons.

    Returns:
        List: A list of formatted ranking results with stability scores.
    """

    # Limit `k_sample` to at most 100
    if k_sample is None or k_sample > 100:
        k_sample = 100

    # Generate rankings using randomized sampling from the interest region
    ranking_list = randomized_get_next(interest_region, columns, num_of_rankings, num_ret_tuples, num_of_samples, k_sample)

    results = []

    for rank in ranking_list:
        angle, stability = rank

        # Convert the angle to a weight vector
        ranking_function = from_angle_to_vector(angle)

        # Compute the ranking based on the weight vector
        ranking = find_ranking(ranking_function, columns).reset_index(drop=True)

        # Select only the top `num_ret_tuples` players with relevant columns
        final_df = ranking.loc[:num_ret_tuples-1, ["user_id", "name", "rank", columns[0], columns[1]]]
        results.append(format_ranking_output(final_df, ranking_function, stability, columns))

    return results

def format_ranking_output(final_df: pd.DataFrame, ranking_function: Tuple[float, float], stability: float, columns: List[str]) -> dict:
    """Formats ranking output into dictionary format using original_df for values."""

    # Select relevant data from original_df using user_ids from final_df
    user_ids = final_df['user_id'].values
    original_data = original_df[original_df['user_id'].isin(user_ids)][['user_id', 'name', *columns]]

    return {
        "ranked_list": original_data.to_dict(orient="records"),
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

    # Computes the interest region of ranking functions
    interest_region = compute_interest_region(constraints)

    # Handle sorting based on the selected ranking algorithm
    if algorithm == "raysweeping":
        results = process_raysweeping(interest_region, columns, num_of_rankings, num_ret_tuples)
    elif algorithm == "randomized-rounding":
        results = process_randomized_rounding(interest_region, columns, num_of_rankings, num_ret_tuples, num_of_samples, k_sample)
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

def calculate_ordering_exchange_angle(interest_region: Region, item_indices: Tuple[int,int], columns: List[str]) -> float | None:
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
    if (interest_region.theta_min <= angle) and (angle <= interest_region.theta_max):
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
        
    return float(angle)
    
def compute_raysweeping_heap(interest_region: Region, columns: List[str]) -> List[Tuple[float, Region]]:
    """
    Computes the raysweeping heap of ranking regions based on the given angle region and columns.

    Args:
        region (Region): A tuple representing the angle region (min_theta, max_theta).
        columns (List[str]): A list of column names used for ranking.

    Returns:
        List[Tuple[float, Region]: A max heap containing ranking regions and their stability.
    """
    print(f"Interest region: {interest_region}")

    # Find the initial ranking based on the minimum angle in the given region
    init_rank = find_ranking(from_angle_to_vector(interest_region.theta_min), columns)

    min_heap = [] # Heap to store ordering exchange angles for adjacent item pairs

    # Calculate ordering exchange angles for each adjacent pair of items
    for i in range(len(init_rank)-1):
        angle = calculate_ordering_exchange_angle(interest_region, [i, i+1], columns)
        if angle is not None:
            # Add valid angles to the min heap with their corresponding indices
            heapq.heappush(min_heap, (angle, [i, i+1]))

    old_angle = interest_region.theta_min 
    
    max_heap = [] # Max heap to store ranking regions and their stability

    # Calculate the range area of the angle region
    range_area = interest_region.theta_max - old_angle 

    # Process the min heap to calculate stability for each angle
    while len(min_heap) > 0:
        # Get the smallest angle and its indices
        angle, indices = heapq.heappop(min_heap)
        index_1, index_2 = indices

        # Calculate stability and store it in the max heap
        stability = (angle - old_angle) / range_area  
        heapq.heappush(max_heap, (-stability, Region(old_angle, angle)))  

        old_angle = angle  
    
    return max_heap

def generate_randomized_angles(interest_region: Region, num_of_smaples : int) -> List[float]:
    res  = []
    for _ in range(num_of_smaples):
        angle = float(np.random.uniform(interest_region.theta_min, interest_region.theta_max))
        res.append(angle)
    
    return res

def randomized_get_next(interest_region: Region, columns: List[str], num_of_rankings: int, num_ret_tuples: int, num_of_samples: int, k_sample: int) -> List[Tuple[float, float]]:
    """
    Randomly samples weight vectors within the given interest_region, 
    computes rankings based on those vectors, 
    and returns a list of the most frequently occurring rankings along with their associated stability scores.
    """

    # Draws `num_of_samples` samples from the interest region
    samples = generate_randomized_angles(interest_region, num_of_samples)

    ranking_counts = {} # Dict with ranking_id as key and the number of times it was samples as value
    ranking_angles = {} # Dict with ranking_id as key and angle created by it's weight vector as value

    # Compute ranking for each sampled weight vector
    for sample in samples:
        # Computes the ranking of the items that corresponds to the sampled weight function
        ranking = find_ranking(from_angle_to_vector(sample), columns)

        # Extract top `k_sample` users from ranking
        ranking = ranking.iloc[:k_sample]['user_id']

        # Generate a unique ID for this ranking
        ranking_id = generate_ranking_id(ranking)

        ranking_angles[ranking_id] = sample

        # Checks if the ranking was previously discovered, and sets it's counter accordingly
        if ranking_id in ranking_counts:
            ranking_counts[ranking_id] += 1
        else:
            ranking_counts[ranking_id] = 1

    ranking_list = []

    # If not specified, return all discovered rankings
    if num_of_rankings == None:
        num_of_rankings = len(ranking_counts)

    for _ in range(num_of_rankings):
        try:
            max_ranking = max(ranking_counts, key=ranking_counts.get) # Find the most frequent ranking
        except ValueError:
            break # No more rankings left

        angle = ranking_angles[max_ranking]
        stability = ranking_counts[max_ranking] / num_of_samples
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

def get_columns_names() -> List[str]:
    """Returns a list of column names in the cleaned DataFrame"""

    return cleaned_df.columns.tolist()

def sample_first_five_entries():
    """Returns the first five entries from the dataset."""
    
    return original_df.head(5).replace({np.nan: None}).to_dict(orient="records")

