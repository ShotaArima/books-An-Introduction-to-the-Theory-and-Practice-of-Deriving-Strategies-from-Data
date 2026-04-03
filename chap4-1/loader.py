import pandas as pd

import os
import requests

class DataLoader:
    def __init__(self, save_path: str, url: str):
        self.save_path = save_path
        self.url = url

    def data_load(self):
        if os.path.isfile(self.save_path):
            print("file exist")
        else:
            response = requests.get(self.url)

            if response.status_code == 200:
                with open(self.save_path, 'wb') as file:
                    file.write(response.content)
                print("complete download data")
            else:
                print("fault", response.status_code)

    def load(self):
        df = pd.read_csv(self.save_path)
        df.columns = ["index", "speed", "dist"]
        return df["speed"].values, df["dist"].values
