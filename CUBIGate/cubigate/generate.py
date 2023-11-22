
class CubigDPGenerator():
    def __init__(
        self, 
        api: str,
        model_checkpoint: str,
        feature_extractor: str,
        result_folder: str,
        tmp_folder: str,
        modality: str,
        data_folder: str,
        ) -> None:
        pass

    def train(
        self,
        condition_guidance_scale: float,
        inference_steps: int,
        batch_size: int,
        variation_strength: float,
        variation_degree_schedule: str,
        count_threshold: float,
        image_size: str,
        
        num_initial_samples: int,
        initial_prompt: str,
        ):
        pass

    def generate(
        self,
        base_data: str,
        image_size: str,
        num_samples: int,
        suffix: str
    ):
        pass