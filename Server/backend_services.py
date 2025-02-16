### implementation of the backend services from the paper.

import math
import numpy as np
import pandas as pd
import json
import heapq
from fastapi import FastAPI
import os
import hashlib


file_path = os.path.join(os.path.dirname(__file__), "Data", "GM_players_statistics_cleaned.csv")
df = pd.read_csv(file_path)


def generate_ranking_id(ranking):
    """Generates a unique ID for a ranking by hashing the tuple."""
    ranking_str = ",".join(map(str, ranking))  # Convert to string
    return hashlib.md5(ranking_str.encode()).hexdigest()  # Hash it

def sort_data(constraints: list, method: str, columns: list, num_ret_tuples: int, num_of_rankings : int, num_of_smaples : int = None, k_sample=None) -> tuple[pd.DataFrame, float]:

    """
    Sorts the data based on the constraints and method provided.
    Returns the first "num_ret_tuples" rows for the "num_of_ranking" first most stabel rankings, the ranking functions and the stability score.
    """
    # Checking parameters
    if not(columns[0] in df.columns) or not(columns[1] in df.columns) or (len(columns) != 2):
        raise ValueError("Columns not found in the data or invalid number of columns provided")
    
    if len(constraints) != 0:
        region_in_angles = find_feasible_angle_region(constraints)
        if region_in_angles is None:
            raise ValueError("No feasible region found - Infeasible constraints")

    # Sorting the data
    if method == "Ray Sweeping":
        ranking_heap = ray_sweeping(region_in_angles, columns)
        res = []
        for i in range(num_of_rankings):
            stability, angle_range_rank = heapq.heappop(ranking_heap)
            stability = -stability
            angle = np.mean(angle_range_rank)
            ranking = find_ranking(from_angle_to_vector(angle), columns)
            ranking_function = from_angle_to_vector(angle)
            ranking = ranking.reset_index(drop=True)
            final_df = ranking.loc[:num_ret_tuples-1, ["user_id","name","Rank",columns[0], columns[1]]]
            print(final_df)
            final_df_dict = final_df.to_dict(orient="records")
            ranking_function_final = {"w1": ranking_function[0], "w2": ranking_function[1]}
            res.append({"Ranking":final_df_dict, "Ranking_Function":ranking_function_final, "Stability":stability})
            

    elif method == "Randomized Rounding":

        if k_sample==None or k_sample>100:
            k_sample = 100

        ranking_list = randomized_get_next(region_in_angles, columns, num_of_rankings, num_ret_tuples, num_of_smaples, k_sample)
        res =[]

        for rank in ranking_list:
            angle = rank[0]
            stability = rank[1]
            ranking = find_ranking(from_angle_to_vector(angle), columns)
            ranking_function = from_angle_to_vector(angle)
            ranking = ranking.reset_index(drop=True)
            final_df = ranking.loc[:num_ret_tuples-1, ["user_id","name","Rank",columns[0], columns[1]]]
            final_df_dict = final_df.to_dict(orient="records")
            ranking_function_final = {"w1": ranking_function[0], "w2": ranking_function[1]}
            res.append({"Ranking":final_df_dict, "Ranking_Function":ranking_function_final, "Stability":stability})

    else:
        raise ValueError(f"Invalid method provided - {method}, choose from 'Ray_Sweeping' or 'Randomized_Rounding'")
    
    # stability = get_ranking_stability(ranking, columns)
    # final_df = ranking.loc[:num_ret_tuples,["user_id","name","Rank",columns[0], columns[1]]]

    # Convert list to JSON and save it to a file
    with open("res.json", "w") as file:
        json.dump(res, file, indent=4)
    
    return res


def get_ranking_stability(range, columns):
    """
    Returns the stability score of the ranking function.
    """
    # TODO: Implement stability score calculation
    return 1.0
    

def calculate_exchange_ordering_angle(region_in_angles, indexes, columns):
    """
    Calculates the exchange ordering angle for a pair of players.
    """
    item_1 = df.loc[indexes[0], columns]
    item_2 = df.loc[indexes[1], columns]

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
    
    df_ranked = df.copy()
    df_ranked["Rank"] = df_ranked[columns[0]]*waights[0] + df_ranked[columns[1]]*waights[1]
    df_ranked = df_ranked.sort_values(by="Rank", ascending=False)
    return df_ranked


def find_feasible_angle_region(constraints):
    """Finds the intersection of angle constraints given as (a, b, sign).
    """

    def find_angle(a, b):
        return np.degrees(np.arctan2(b, a))

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
    return df.columns.tolist()


def main():

    res = sort_data([(1,2,"<="),(1, 1, ">=")], "Randomized Rounding", ["followers", "bullet_win"], num_of_rankings=2, num_ret_tuples=3, num_of_smaples=9000, k_sample=40)
    # print(res)

    # print(f"Stability score: {res[1]}")


if __name__ == "__main__":
    main()
