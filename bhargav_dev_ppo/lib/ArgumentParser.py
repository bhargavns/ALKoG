from lib.Logger import Logger

g_Logger = Logger(__name__)
print = g_Logger.print

class ArgumentParser:
    def __init__(self):
        self.arguments = None
    """
    setArguments
    sets self.arguments to be the inputArguments provided, assuming inputArguments is a list of = seperated key, value pairs 
    if an argument key is not in required or optional arguments, it will be ignored 
    """
    def setArguments(self, inputArguments: list, required_arguments:list=[], optional_arguments:dict={}):
        "inputArguments is just what typically gets passed to argc, an array"
        self.arguments={}
        for inputArgument in inputArguments:
            if(not inputArgument.count("=")):
                raise Exception(f"Argument passed to setArguments is not = character set. We assume <variable>=<value> input to the script. Failing argument: {inputArgument}")
            (key,value) = (inputArgument.split('=')[0], inputArgument.split('=')[1])    
            if(key not in required_arguments + list(optional_arguments.keys())):
                print(f"Provided argument {key} not in required or optional arguments. Ignoring.")
            else:
                self.set(key, value)
                
        for required_argument in required_arguments:
            if(required_argument not in self.arguments):
                raise Exception(f"Missing {required_argument} from provided inputs")
        for optional_argument in optional_arguments:
            if(optional_argument not in self.arguments):
                print(f"Optional argument {optional_argument} not found in input. Continuing...")
                self.set(optional_argument, optional_arguments[optional_argument])
    
    def get(self, name):
        if(name not in self.arguments):
            raise Exception(f"Cannot find {name} in arguments.")
        return self.arguments[name]
    def set(self,key,value):
        self.arguments[key]=value 
    def printArguments(self):
        for argument in self.arguments:
            print(f"\tKey:{argument}\tValue:{self.arguments[argument]}")
