from cubigate.generate import CubigDPGenerator
from cubigate.dp.utils.arg_utils import str2bool
import argparse
import os

#fixed values for display
generator = CubigDPGenerator()

#Just train the model to make DP-synthetic data
def train_data_generation_model(iterations=2, epsilon=1, delta=0):
    data_checkpoint=generator.train(iterations, epsilon, delta)
    print(data_checkpoint)
    return  data_checkpoint

#Just generate data with your data checkpoint (data checkpoint means model chekcpoint in Cubigate)
#Output: zip file of new data.
def generate_dp_data(base_data="./result/cookie/1/_samples.npz"):
    new_data=generator.generate( base_data)
    print(type(new_data))
    return new_data
    
#Run the entire process of Cubigate (train, generate) => output is zip file of new image data
def train_generate_dp_data(iterations=2, epsilon=1, delta=0):
    data_checkpoint=generator.train(iterations, epsilon, delta)
    new_data=generator.generate(data_checkpoint)
    print(type(new_data))
    return new_data
