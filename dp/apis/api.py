from abc import ABC, abstractmethod
import argparse
from typing import List



class API(ABC):
    def __init__(self, args=None):
        self.args = args
        self._result_folder = None
        self._live = -1
        self._live_loading_target = None
        self._modality = None

    @staticmethod
    def command_line_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--api_help',
            action='help')
        return parser

    @classmethod
    def from_command_line_args(cls, args, result_folder, live_loading_target, modality):
        """
        Creating the API from command line arguments.

        Args:
            args: (List[str]):
            The command line arguments
        Returns:
            API:
                The API object.
        """
        args = cls.command_line_parser().parse_args(args)
        api = cls(**vars(args), args=args)
        api._result_folder = result_folder
        if api._result_folder is not None:
            api._live = 0
        if live_loading_target is not None:
            api._live = 1
            api._live_loading_target = live_loading_target
        api._modality = modality
        return api

    @abstractmethod
    def random_sampling(self, num_samples, size, prompts=None):
        """
        Generates a specified number of random image samples based on a given
        prompt and size.

        Args:
            num_samples (int, optional):
                The number of image samples to generate.
            size (str, optional):
                The size of the generated images in the format
                "widthxheight", e.g., "1024x1024".
            prompts (List[str], optional):
                The text prompts to generate images from. Each promot will be
                used to generate num_samples/len(prompts) number of samples.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x width x height x
                channels] with type np.uint8 containing the generated image
                samples as numpy arrays.
            numpy.ndarray:
                A numpy array with the first dimension equaling to
                num_samples containing additional information such as labels.
        """
        pass

    @abstractmethod
    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree=None, t=None):
        """
        Generates a specified number of variations for each image in the input
        array.

        Args:
            images (numpy.ndarray):
                A numpy array of shape [num_samples x width x height
                x channels] containing the input images as numpy arrays of type
                uint8.
            additional_info (numpy.ndarray):
                A numpy array with the first dimension equaling to
                num_samples containing additional information such as labels or
                prompts provided by image_random_sampling.
            num_variations_per_image (int):
                The number of variations to generate for each input image.
            size (str):
                The size of the generated image variations in the
                format "widthxheight", e.g., "1024x1024".
            variation_degree (int or float, optional):
                The degree of image variation.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x num_variations_per_image
                x width x height x channels] containing the generated image
                variations as numpy arrays of type uint8.
        """
        pass
        
        @abstractmethod
        def _live_save(self, samples: List, additional_info, prefix: str):
            pass

        @abstractmethod
        def _live_load(self, path: str):
            pass
            
