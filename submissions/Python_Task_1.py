#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

def generate_car_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'id_1', 'id_2', and 'car'.

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    
    # To pivot the dataframe
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')
    # To replace null values with 0
    car_matrix = car_matrix.fillna(0)
    # To set diagonal values to 0
    for index in car_matrix.index:
        car_matrix.at[index, index] = 0
    return car_matrix

# Read CSV file before calling the function
df = pd.read_csv('dataset-1.csv')
car_matrix = generate_car_matrix(df)
car_matrix


# In[6]:


def get_type_count(df: pd.DataFrame) -> dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame): Input DataFrame with a column 'car'.

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Add a new categorical column 'car_type' based on 'car' values
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts
# Read CSV file before calling the function
df = pd.read_csv('dataset-1.csv')
sorted_counts = get_type_count(df)
sorted_counts


# In[7]:


def get_bus_indexes(df: pd.DataFrame) -> list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame): Input DataFrame with a column 'bus'.

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Calculate the mean value of the 'bus' column
    mean_bus = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sort the list in ascending order
    bus_indexes.sort()

    return bus_indexes
# Read CSV file before calling the function
df = pd.read_csv('dataset-1.csv')
bus_indexes = get_bus_indexes(df)
bus_indexes


# In[8]:


def filter_routes(df: pd.DataFrame) -> list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'route' and 'truck'.

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Group by 'route' and calculate the mean of 'truck' for each group
    route_truck_mean = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    filtered_routes = route_truck_mean[route_truck_mean > 7].index.tolist()

    # Sort the list in ascending order
    filtered_routes.sort()

    return filtered_routes
# Read CSV file before calling the function
df = pd.read_csv('dataset-1.csv')
filtered_routes = filter_routes(df)
filtered_routes


# In[15]:


def multiply_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame): Input DataFrame representing a matrix.

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
        
    # Apply custom conditions to multiply values
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

#To create a copy of dataframe to avoid modifying original
modified_matrix = car_matrix.copy()
modified_matrix = multiply_matrix(modified_matrix)
modified_matrix


# In[ ]:




