# Assignment 1
---

## Running the code
The code can be run with `python Assignment1.py`. The file assumes you run the code from the root of the repository, however it takes an optional argument `--data_path` to specify the path to the data_folder. You can also specify the `--batch_size`, but the default is 16.

## Task 1
I chose to go for the RestNet18 model, as this model seems fitting for the task goal and scale. For 
the loss function I went with the AdamW optimizer, as it is a good general optimizer for most tasks.
For calculating the loss I went with the CrossEntropyLoss.


