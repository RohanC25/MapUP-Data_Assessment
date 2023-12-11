#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import networkx as nx

def calculate_cumulative_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df, using NetworkX.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: Cumulative distance matrix
    """
    # Create a NetworkX graph from the DataFrame
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_edge(row["id_start"], row["id_end"], weight=row["distance"])

    # Calculate the shortest paths between all pairs of nodes in the graph
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))

    # Initialize a dictionary to store the cumulative distances
    cumulative_distances = {}

    # Iterate over each start node
    for start_id, end_distances in shortest_paths.items():
        cumulative_distances[start_id] = {}  # Initialize inner dictionary for the start node

        # Iterate over each end node and its distance from the start node
        for end_id, distance in end_distances.items():
            # Initialize the cumulative distance to 0
            cumulative_distance = 0

            # Iterate over the shortest path between the start and end nodes
            shortest_path_nodes = nx.shortest_path(G, source=start_id, target=end_id)
            for i in range(len(shortest_path_nodes) - 1):
                intermediate_node = shortest_path_nodes[i]
                next_node = shortest_path_nodes[i + 1]

                # Add the weight of the edge between the current and next nodes to the cumulative distance
                cumulative_distance += G.edges[(intermediate_node, next_node)]["weight"]

            # Store the cumulative distance for the current pair of start and end nodes
            cumulative_distances[start_id][end_id] = cumulative_distance

    # Convert the cumulative distances dictionary to a Pandas DataFrame
    cumulative_distances_df = pd.DataFrame(cumulative_distances)

    # Set the index and column names of the DataFrame
    cumulative_distances_df.index.names = ["id_start"]
    cumulative_distances_df.columns.names = ["id_end"]

    # Fill missing values with 0 (diagonal elements)
    cumulative_distances_df.fillna(0, inplace=True)

    return cumulative_distances_df

# Read csv file before calling function
df = pd.read_csv('dataset-3.csv')
cumulative_distances_result = calculate_cumulative_distances(df)
cumulative_distances_result


# In[10]:


def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): Input DataFrame representing a distance matrix.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Reset the index to convert the 'id_start' index to a column
    df_unrolled = df.reset_index()

    # Melt the DataFrame to convert it from wide to long format
    df_unrolled = pd.melt(df_unrolled, id_vars='id_start', var_name='id_end', value_name='distance')

    # Remove rows where 'id_start' is equal to 'id_end'
    df_unrolled = df_unrolled[df_unrolled['id_start'] != df_unrolled['id_end']]

    # Reset the index of the resulting DataFrame
    df_unrolled.reset_index(drop=True, inplace=True)

    return df_unrolled

df_unrolled = unroll_distance_matrix(cumulative_distances_result)

# Print the resulting DataFrame
df_unrolled


# In[56]:


def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): Input DataFrame containing columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): The reference ID for which to find IDs within the specified percentage threshold.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter rows for the reference value
    reference_rows = df[df['id_start'] == reference_id]
    
    # Calculate the average distance
    avg_distance = reference_rows['distance'].mean()
    
    # Calculate the threshold range (10% of the average distance)
    threshold_range = 0.1 * avg_distance
    
    # Find rows within the threshold range
    within_threshold_rows = df[
        (df['distance'] >= avg_distance - threshold_range) &
        (df['distance'] <= avg_distance + threshold_range)
    ]
    
    # Create a DataFrame from the result data
    result_df = pd.DataFrame({
        'id_start': within_threshold_rows['id_start'],
        'id_end': within_threshold_rows['id_end'],
        'distance': within_threshold_rows['distance'],
        'avg_distance': avg_distance,
        'threshold_range': avg_distance * 0.1
    })

result_df


# In[22]:


def calculate_toll_rate(df_unrolled: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the input DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'id_start', 'id_end', 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with added columns for each vehicle type and their respective toll rates.
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Add columns for each vehicle type with their respective toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df_unrolled['distance'] * rate_coefficient

    return df

# Usage:
unrolled_distances_with_rates = calculate_toll_rate(df_unrolled)
unrolled_distances_with_rates


# In[ ]:


import datetime
def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): Input DataFrame containing columns 'id_start', 'id_end', 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with added columns 'start_day', 'start_time', 'end_day', 'end_time', and 'toll_rate'.
    """
    # Define time ranges and discount factors
    time_ranges_weekdays = [
        ((0, 0, 0), (10, 0, 0), 0.8),
        ((10, 0, 0), (18, 0, 0), 1.2),
        ((18, 0, 0), (23, 59, 59), 0.8)
    ]

    time_ranges_weekends = [
        ((0, 0, 0), (23, 59, 59), 0.7)
    ]

    # Create a DataFrame to store the calculated toll rates
    toll_rates_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 'toll_rate'])

    # Iterate over each unique (id_start, id_end) pair
    for _, group in df.groupby(['id_start', 'id_end']):
        # Create an empty DataFrame to store the calculated toll rates for each day and time range
        rates_by_day = pd.DataFrame(columns=['id_start', 'id_end', 'distance','start_day', 'start_time', 'end_day', 'end_time', 'toll_rate'])

        # Iterate over each day of the week
        for day in range(7):
            # Choose the appropriate time ranges based on weekdays or weekends
            if day < 5:  # Weekdays
                time_ranges = time_ranges_weekdays
            else:       # Weekends
                time_ranges = time_ranges_weekends

            # Iterate over each time range
            for start_time, end_time, discount_factor in time_ranges:
                start_datetime = datetime.datetime.combine(datetime.date.today(), datetime.time(*start_time))
                end_datetime = datetime.datetime.combine(datetime.date.today(), datetime.time(*end_time))

                # Calculate the toll rate for the current time range
                toll_rate = group['distance'] * discount_factor

                # Add the calculated toll rate to the DataFrame
                rates_by_day = rates_by_day.append({
                    'id_start': group['id_start'].values[0],
                    'id_end': group['id_end'].values[0],
                    'distance': group['distance'].values[0],
                    'start_day': day,
                    'start_time': start_datetime.time(),
                    'end_day': day,
                    'end_time': end_datetime.time(),
                    'toll_rate': toll_rate.mean()  # You can adjust this depending on your calculation logic
                }, ignore_index=True)

        # Merge the calculated toll rates back to the main DataFrame
        toll_rates_df = pd.concat([toll_rates_df, rates_by_day], ignore_index=True)

    return toll_rates_df

# Example usage:

result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_df)

# Display the resulting DataFrame
print(result_with_time_based_toll_rates.head())

