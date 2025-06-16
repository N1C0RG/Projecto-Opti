import os

class DataSaver:
    def __init__(self):
        pass

    def save_data(self, filename : str, data : list):
        """
        Save data to a CSV file.
        - filename: path to the csv file (relative to 'data' folder)
        - data: nested lists (2D or 3D) to save
        """
        with open(os.path.join('data', filename), 'w', newline='') as file:
            for row in data:
                file.write(row + '\n')