# Assignment 1
---

## Running the code
The code can be run with `python Assignment1.py`. The file assumes you run the code from the root of
the repository, however it takes an optional argument `--data_path` to specify the path to the
data_folder. You can also specify the `--batch_size`, but the default is 16.

Just running the code will run everything, including the grid search. The grid search can be time consuming, so if you want to skip it, you can set the `--skip_grid_search` flag. This will directly load the model from the `models` folder.

If you want to run a specific task you can add the flag `--task <task-number>`.

## Task 1
I chose to go for the RestNet18 model, as this model seems fitting for the task goal and scale. For 
calculating the loss I went with the CrossEntropyLoss.

I chose to use a GridSearch algorithm to find the best hyperparameters for the model. The 
hyperparameters I chose to tune was learning rate, optimizer and whether to transform the images by 
randomly flipping/rotating or not.

I ran the grid search with the two optimizers: AdamW and SGD as well as the learning rates
[1e-4, 1e-5] and either with or without transformations, with a 50% chance of transforming 
every image. I ran all the runs with 20 epochs, but with early stopping with a patience score of 5. 
Therefore most of the runs ended early. When the validation loss did not increase for 5 epochs, the 
last best model was chosen. The grid search output file is included as `grid_search.out`.

