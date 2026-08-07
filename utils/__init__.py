"""Core library for the Robo Data Scientist AutoML platform.

Module responsibilities, in the order the pipeline uses them:

- :mod:`utils.constants`        environment-driven configuration
- :mod:`utils.logging_utils`    centralised logging setup
- :mod:`utils.validation`       input checks applied before any work is done
- :mod:`utils.data_utils`       dataset loading, target detection and cleaning
- :mod:`utils.task_inference`   classification vs. regression inference
- :mod:`utils.feature_engineer` preprocessing transformers and pipeline builder
- :mod:`utils.model_trainer`    model search, evaluation and the leaderboard
- :mod:`utils.model_artifact`   the train/serve serialization contract
- :mod:`utils.predict`          inference from a saved artifact
"""
