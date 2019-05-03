To install the needed dependencies use:
 
 `
 pip install -r requirements.txt
 `
 
 To get matplot lib to work on mac replace `python` with `pythonw`
 
 
 To run experiments use commands like:
 
 ```
 pythonw -m experiments.mgda_model_train
 pythonw -m experiments.coordinate_model_train
 ```
 
 
 To run visualizations use commands like:
 
 ```
 pythonw -m visualizations.dynamics_visualization with experiment_nums=2,3 algo_labels=Baseline,Ours tile_labels=Normal,Pit
 pythonw -m visualizations.value_function_visualization with experiment_nums=2,3 labels=Ours,Baseline show_art=True
 pythonw -m visualizations.loss_visualization with experiment_num=2
 ```
 
 
 
 All output can be found in the logs directory by default and the generated images are in logs/generated_images
