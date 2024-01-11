from cubigate.generate import CubigDPGenerator
from main import *
import io

import bentoml
from bentoml.io import JSON, File, Multipart
from pydantic import BaseModel


svc = bentoml.Service("dp_msv", runners=[])

class generate_input(BaseModel):
    base_data: str
    
class train_input(BaseModel):
    iterations: int = 2
    epsilon: int = 1
    delta: int = 0

@svc.api(input=JSON(pydantic_model=generate_input), output=File())
def generate(input_data: generate_input):
    base_data = input_data.base_data
    output_file_path = generate_dp_data(base_data).filename
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
    
    base_data = train_data_generation_model(iterations, epsilon, delta)
    res = {
        "base_data": base_data
    }
    
    return res
################################################################################################

@svc.api(input=JSON(), output=JSON())
def init_train(input_data: train_input):
    iterations = input_data.iterations
    epsilon = input_data.epsilon
    delta = input_data.delta
    
    output_file_path = initialize_training(iterations, epsilon, delta)
    res = {
        "initial": output_file_path
    }
    return res


@svc.api(input=JSON(), output=JSON())
def variate(input_data):
    previous = input_data.get("previous", "")
    
    packed_samples = variate_prev_data(previous)
    res = {
        "packed_samples": packed_samples
    }
    return res


@svc.api(input=JSON(), output=JSON())
def measure(input_data):
    variated = input_data.get("variated", "")
    epsilon = input_data.get("epsilon", 1)
    delta = input_data.get("delta", 0)
    
    output_file_path = measure_variated(variated, epsilon, delta)
    
    res = {"estimated_distribution": output_file_path}
    return res


@svc.api(input=JSON(), output=JSON())
def select(input_data):
    measured = input_data.get("measured", "")
    variated = input_data.get("variated", "")

    output_file_path = select_measured(measured, variated)
    res = {"selected_samples": output_file_path}
    
    return res