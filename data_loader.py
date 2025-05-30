import pandas as pd
import csv
import os
import time

class DataLoader:
    def __init__(self):
        pass

    def load_data(self, filename, axes, value_col):
        """
        General loader for 2D or 3D CSVs.
        - filename: path to the csv file (relative to 'data' folder)
        - axes: list of column names to use as axes (e.g., ['Sector', 'Especialidad', 'Dia'])
        - value_col: column name for the value (e.g., 'ceit')
        Returns: nested lists (2D or 3D)
        """
        data = []
        with open(os.path.join('data', filename), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                entry = [int(row[ax]) for ax in axes]
                entry.append(int(row[value_col]))
                data.append(entry)

        # Find max index for each axis
        max_indices = [max(row[i] for row in data) for i in range(len(axes))]

        # Initialize nested lists
        def nested_list(levels):
            if len(levels) == 1:
                return [0 for _ in range(levels[0])]
            return [nested_list(levels[1:]) for _ in range(levels[0])]

        matrix = nested_list(max_indices)

        # Fill the matrix
        for row in data:
            idx = [i-1 for i in row[:-1]]  # 0-based
            val = row[-1]
            if len(idx) == 2:
                matrix[idx[0]][idx[1]] = val
            elif len(idx) == 3:
                matrix[idx[0]][idx[1]][idx[2]] = val
            else:
                raise ValueError("Only 2D, 3D or 4D supported.")

        return matrix
        

'''
dl = DataLoader()

ceit = dl.load_data('ceit.csv', ['Sector', 'Especialidad', 'Dia'], 'ceit')

for( i, sector) in enumerate(ceit):
    for (e, especialidad) in enumerate(sector):
        for (d, ceit_value) in enumerate(especialidad):
            print(f"Sector {i+1}, Especialidad {e+1}, Dia {d+1}: CEIT = {ceit_value}")


jet = dl.load_data('jet.csv', ['Especialidad', 'Dia'], 'jet')
for (e, especialidad) in enumerate(jet):
    for (d, jet_value) in enumerate(especialidad):
        print(f"Especialidad {e+1}, Dia {d+1}: JET = {jet_value}")
'''