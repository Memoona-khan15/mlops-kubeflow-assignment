# pipeline.py
import kfp
from kfp import dsl
from kfp.components import load_component_from_file

data_extraction_op = load_component_from_file('components/data_extraction.yaml')
preprocess_op = load_component_from_file('components/preprocessing.yaml')
training_op = load_component_from_file('components/training.yaml')
evaluation_op = load_component_from_file('components/evaluation.yaml')

@dsl.pipeline(
    name='diabetes-regression-pipeline',
    description='A pipeline that trains RandomForest on the diabetes dataset.'
)
def my_pipeline(data_path: str = '/workspace/data/raw_data.csv'):
    extract = data_extraction_op(data_path=data_path, out_path='/tmp/data.csv')
    preprocess = preprocess_op(data_path=extract.output, train_path='/tmp/train.csv', test_path='/tmp/test.csv')
    train = training_op(train_path=preprocess.outputs['train_path'], model_path='/tmp/model.joblib')
    eval = evaluation_op(test_path=preprocess.outputs['test_path'], model_path=train.outputs['model_path'], metrics_path='/tmp/metrics.json')

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(my_pipeline, 'pipeline.yaml')
    print("Compiled pipeline.yaml")
