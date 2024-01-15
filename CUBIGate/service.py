from cubigate.generate import CubigDPGenerator
from main import *
import io

import bentoml
from bentoml.io import JSON, File, Multipart
from pydantic import BaseModel


svc = bentoml.Service("dp_msv", runners=[])

class generate_input(BaseModel):
    data_checkpoint_path: str
    
class train_input(BaseModel):
    iterations: int = 2
    epsilon: int = 1
    delta: int = 0

@svc.api(input=JSON(pydantic_model=generate_input), output=File())
def generate(input_data: generate_input):
    data_checkpoint_path = input_data.data_checkpoint_path
    output_file_path = generate_dp_data(data_checkpoint_path).filename
    file_content = None
    with io.open(output_file_path, 'rb') as file:
        file_content = file.read()
    return file_content


# input_spec = Multipart(iterations=int, epsilon=int, delta=int)


@svc.api(input=JSON(pydantic_model=train_input), output=JSON())
def train(input_data: train_input):
    # iterations=2, epsilon=1, delta=0
    iterations = input_data.iterations
    epsilon = input_data.epsilon
    delta = input_data.delta
    
    data_checkpoint_path = train_data_generation_model(iterations, epsilon, delta)
    res = {
        "data_checkpoint_path": data_checkpoint_path
    }
    
    return res
################################################################################################

@svc.api(input=JSON(), output=JSON())
def init_train(input_data):   
    output_file_path = initialize_training()
    res = {
        "initial_sample_path": output_file_path
    }
    return res


@svc.api(input=JSON(), output=JSON())
def variate(input_data):
    initial_sample_path = input_data.get("initial_sample_path", "")
    result_sample_path = variate_prev_data(initial_sample_path)
    res = {
        "result_sample_path": result_sample_path
    }
    return res


@svc.api(input=JSON(), output=JSON())
def measure(input_data):
    result_sample_path = input_data.get("result_sample_path", "")
    epsilon = input_data.get("epsilon", 1)
    delta = input_data.get("delta", 0)
    
    output_file_path = measure_variated(result_sample_path, epsilon, delta)
    
    res = {"estimated_distribution_path": output_file_path}
    return res


@svc.api(input=JSON(), output=JSON())
def select(input_data):
    estimated_distribution_path = input_data.get("estimated_distribution_path", "")
    result_sample_path = input_data.get("result_sample_path", "")

    output_file_path = select_measured(estimated_distribution_path, result_sample_path)
    res = {"data_checkpoint_path": output_file_path}
    
    return res