import bentoml
from bentoml.exceptions import NotFound
import diffusers

if __name__ == "__main__":
    stage1_signatures = {
        "__call__": {"batchable": False},
        "encode_prompt": {"batchable": False},
    }

    stage1_model_tag = "runwayml/stable-diffusion-v1-5"
    try:
        bentoml.diffusers.import_model(
            "runwayml-stable_defussion:v1.5",  # model tag in BentoML model store
            stage1_model_tag,
            pipeline_class=diffusers.StableDiffusionPipeline# huggingface model name
            )
    except NotFound:
        bentoml.diffusers.import_model(
            stage1_model_tag, "runwayml/stable-diffusion-v1-5",
            signatures=stage1_signatures,
            #variant="fp16",
            pipeline_class=diffusers.StableDiffusionPipeline,)

    
    stage2_model_tag = "compVis-stable_defussion:v1.4"
    try:
        bentoml.diffusers.import_model(
            stage2_model_tag, "CompVis/stable-diffusion-v1-4",
            #variant="fp16",
            pipeline_class=diffusers.StableDiffusionImg2ImgPipeline
        )
    except NotFound:
        bentoml.diffusers.import_model(
            stage2_model_tag, "CompVis/stable-diffusion-v1-4",
            #variant="fp16",
            pipeline_class=diffusers.StableDiffusionImg2ImgPipeline
        )

