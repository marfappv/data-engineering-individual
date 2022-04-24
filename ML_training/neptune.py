#!pip3 install neptune-client
#!pip3 install neptune-cli --upgrade

import neptune.new as neptune

run = neptune.init(
    project="marfappv/data-eng-ind",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNGJmODc1MC0yMWJmLTQ0ZDAtYjAzOC02NTdhN2RlNTE0YzEifQ==",
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()

# Run the file:
# python3 neptune.py