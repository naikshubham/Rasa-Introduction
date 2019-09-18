from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

# create args dictionary
args = {"pipeline" : "spacy_sklearn"}

# create a configuration and trainer
config = RasaNLUConfig(cmdline_args = args)
trainer = Trainer(config)

# Load the training data
training_data = load_data("./training_data.json")

# create an interpreter by training the model
interpreter = trainer.train(training_data)

# Test the interpreter
print(interpreter.parse("I'm looking for a Mexican restaurant in the North of town"))