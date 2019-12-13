import argparse
import re
import json


class Config:
    def __init__(self, definition_path: str):
        arg_parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        with open(definition_path) as definition_file:
            definition = json.loads(definition_file.read())
            for key, item in definition.items():
                arg_parser.add_argument(
                    "--" + re.sub("_", "-", key),
                    type=eval(item["type"]),
                    default=item["default"],
                    help=item["help"],
                )

        args = arg_parser.parse_args()
        self.config = vars(args)

    def __getitem__(self, key):
        return self.config[key]
