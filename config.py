import json


class Config:
    def __init__(self, filename: str):
        self.load_from_file_(filename)

    def load_from_file_(self, filename: str):
        file = open(filename)
        self.config = json.loads(file.read())
        file.close()

    def update_(self, config_dict: dict):
        for key in config_dict.keys():
            if key in self.config:
                self.config[key] = config_dict[key]

    def __getitem__(self, key: str):
        return self.config[key]

    def __str__(self):
        return json.dumps(self.config, indent=4)
