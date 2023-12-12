from main import *

train_data_generation_model(iterations=3)
generate_dp_data()
train_generate_dp_data(epsilon=0.5, delta=0.1)
