# Demo

## Prompts

### Visualization

```
Create a flyte script that downloads the dataset at https://github.com/plotly/datasets/blob/master/timeseries.csv and creates a visualization in plotly and run it remotely. The flyte script should use flyte.report to render the visualization.
```

### Model training

```
Run a flyte script that performs hyperparameter optimization trying a random forest model on the penguins data. Assess f1 score as the evaluation metric, and visualize the results using flyte.report.
```


### Batch inference

```
Run a flyte script that embeds the "review" column of the "scikit-learn/imdb" huggingface dataset using the "answerdotai/ModernBERT-base" model. Save the embeddings to a json file using flyte.io.File
```
