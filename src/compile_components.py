# src/compile_components.py
from kfp.components import create_component_from_func
from pipeline_components import data_extraction, preprocess, train_model, evaluate_model

create_component_from_func(data_extraction, base_image='python:3.9', output_component_file='../components/data_extraction.yaml')
create_component_from_func(preprocess, base_image='python:3.9', output_component_file='../components/preprocessing.yaml')
create_component_from_func(train_model, base_image='python:3.9', output_component_file='../components/training.yaml')
create_component_from_func(evaluate_model, base_image='python:3.9', output_component_file='../components/evaluation.yaml')
print("Components compiled to components/*.yaml")
