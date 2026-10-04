"""Strict key=value arguments shared by the standalone scripts."""
from shared.Logger import Logger


class ArgumentParser:
    def __init__(self):
        self.arguments = {}

    def setArguments(self, inputArguments, required_arguments=None, optional_arguments=None):
        required = list(required_arguments or ())
        defaults = dict(optional_arguments or {})
        self.arguments = defaults.copy()
        supplied = set()
        for argument in inputArguments:
            if "=" not in argument:
                raise ValueError(f"Expected key=value, got {argument!r}")
            key, value = argument.split("=", 1)
            if key not in defaults and key not in required:
                raise ValueError(f"Unknown argument: {key}")
            if key in supplied:
                raise ValueError(f"Duplicate argument: {key}")
            supplied.add(key)
            self.arguments[key] = value
        missing = set(required) - supplied
        if missing:
            raise ValueError(f"Missing required arguments: {sorted(missing)}")

    def get(self, name):
        return self.arguments[name]

    def set(self, key, value):
        self.arguments[key] = value

    def printArguments(self):
        logger = Logger(__name__)
        for key, value in self.arguments.items():
            logger.print(f"{key}={value}")
