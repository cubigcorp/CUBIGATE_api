from typing import List
import numpy as np
import logging

TAG = 'TAG'
AUTO = 'AUTO'

class PromptGenerator:
    def __init__(self,
                 initial_prompt: str,
                 args: List,
                 rng: np.random.Generator) -> None:
        self._base = initial_prompt
        self.n_tags = self._base.count(TAG)
        self._args = args
        self._rng = rng
        self._construct()

    def generate(self, batch: int) -> List[str]:
        gen_prompts = [self._base] * batch
        for _ in range(batch):
            for tag, prompts in self.tag_prompts.items():
                if isinstance(prompts, str):
                    raise "Not yet implemented"
                else:
                    selected = prompts[self._choose_random_prompts(tag, batch)]
                    gen_prompts = [g.replace(f'[{TAG}_{tag}]', s) for g, s in zip(gen_prompts, selected)]
        return gen_prompts
                    


    def _construct(self) -> None:
        assert len(self._args) == self.n_tags * 2, "There are undefined tags in the initial prompt."
        self._parse_args()



    def _parse_args(self) -> None:
        self.tag_prompts = {}
        for i in range(0, len(self._args), 2):
            tag = '_'.join(self._args[i].split('_')[2:])
            if self._args[i+1] == AUTO:
                self.tag_prompts[tag] = AUTO
                logging.info(f'Automatically generate prompts for {tag}')
            else:
                self.tag_prompts[tag] = np.array(self._args[i+1].split(','))
                logging.info(f'Possible prompts for {tag}: {self.tag_prompts[tag]}')


    def _choose_random_prompts(self, tag: str, batch: int) -> List[int]:
        return self._rng.choice(np.arange(len(self.tag_prompts[tag])), batch, replace=True)
