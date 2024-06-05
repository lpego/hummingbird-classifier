The Streamlit Graphical User Interface has been written by Aron Eggenberger <aron.eggenberger@wsl.ch> as part of the Civil Service secondment at WSL. 

# How to check which variables to pass to each script
Using `scripts\main_classification_finetune.py` as an example, you can see which arguments the script expects by navigating to the block
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="path to config file with per-script args",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="path with images for training",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        required=True,
        help="path to where to save model checkpoints",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
```
Looking at the first `parser.addArgument` block, this means is that the script expects a variable called `config_file`, that should be of type `str`, and that this is required. 
The last `parser.addArgument` block looks for a variable called `verbose`, it doesn't specify a type so it defaults to boolean, and it doesn't have the `required=True` flag so it is optional.

**TL;DR:** we want in the streamlit app only the variables that each script expects, and we want to check before running only for the required ones. 

# ToDo
 - ~~[] file selector instead of folder selector for config file~~
 - [x] parse files/directories available for model selector
 - [x] make .bat and .sh files to launch streamlit interface
 - [] check if threshold has been modified from default before running
 - [] BUG: required fields checks do not work after clicking Clear
 - [] check modes of redirecting terminal output in case of errors (or success also)

## app_main_classification.py
 - [] refactor functions with current imports
 - [] rename variables (see app_main_assessment.py as example)
  -[] add logging and debug info
 - [] add clear button

## app_preprocess_convert_videos.py
 - [] refactor functions with current imports
 - [] rename variables (see app_main_assessment.py as example)
  -[] add logging and debug info
 - [] add clear button

## app_preprocess_prepare_learning_sets_frames.py
 - [] refactor functions with current imports
 - [] rename variables (see app_main_assessment.py as example)
  -[] add logging and debug info
 - [] add clear button

## app_score_inference.py
 - [] refactor functions with current imports
 - [] rename variables (see app_main_assessment.py as example)
  -[] add logging and debug info
 - [] add clear button