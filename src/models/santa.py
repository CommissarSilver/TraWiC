import inspect
import logging
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.model import InfillModel

logger = logging.getLogger("model")


class SantaCoder(InfillModel):
    """
    interface for interacting with the SantaCoder model
    """

    def __init__(self):
        # @TODO: #2 move these to a config file
        self.FIM_PREFIX = "<fim-prefix>"
        self.FIM_MIDDLE = "<fim-middle>"
        self.FIM_SUFFIX = "<fim-suffix>"
        self.FIM_PAD = "<fim-pad>"
        self.ENDOFTEXT = "<|endoftext|>"

        checkpoint = "bigcode/santacoder"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        self.ENDOFTEXT,
                        self.FIM_PREFIX,
                        self.FIM_MIDDLE,
                        self.FIM_SUFFIX,
                        self.FIM_PAD,
                    ],
                    "pad_token": self.ENDOFTEXT,
                }
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                revision="comments",
                trust_remote_code=True,
                max_length=200,
            ).to(self.device)
            logger.info(f"SantaCoder model successfuly loaded")
        except Exception as e:
            logger.exception(f"Error in loading the SantaCoder model")
            raise Exception("Problem in initializing SantaCoder Model")

    def predict(self, input_text: str) -> str:
        """
        Generate code snippet from the input text

        Args:
            input_text (str): input code. not tokenized.

        Raises:
            e: any error in generating code snippet

        Returns:
            str: geenrated code snippet
        """
        try:
            inputs: torch.Tensor = self.tokenizer.encode(
                input_text, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs: torch.Tensor = self.model.generate(inputs)

            logger.debug(
                f"SantaCoder Invoked - input = ( {input_text} ) - output = ( {self.tokenizer.decode(outputs[0])} )"
            )
            return self.tokenizer.decode(outputs[0])
        except Exception as e:
            logger.exception(f"Error in generating code snippet from SantaCoder")
            raise e

    def extract_fim_part(self, s: str):
        """
        Find the index of <fim-middle>

        Args:
            s (str): input string

        Raises:
            e: any excepetion

        Returns:
            _type_: fim part of the input string
        """
        try:
            start = s.find(self.FIM_MIDDLE) + len(self.FIM_MIDDLE)
            stop = s.find(self.ENDOFTEXT, start) or len(s)
            return s[start:stop]

        except Exception as e:
            logger.exception(f"Error in extracting fim part from SantaCoder output")
            raise e

    def infill(
        self,
        prefix_suffix_tuples: Tuple[str, str, str, str],
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        """
        Generate code snippets by infilling between the prefix and suffix.

        Args:
            prefix_suffix_tuples (_type_): a tuple of form (prefix, suffix)
            max_tokens (int, optional): maximum tokens for the model. Defaults to 200.
            temperature (float, optional): model temp. Defaults to 0.8.
            top_p (float, optional): top_p. Defaults to 0.95.

        Returns:
            str: infilled code snippet
        """
        output_list = True
        if type(prefix_suffix_tuples) == tuple:
            prefix_suffix_tuples = [prefix_suffix_tuples]
            output_list = False

        prompts = [
            f"{self.FIM_PREFIX}{prefix}{self.FIM_SUFFIX}{suffix}{self.FIM_MIDDLE}"
            for infill_obj, prefix, suffix, level in prefix_suffix_tuples
        ]
        # `return_token_type_ids=False` is essential, or we get nonsense output.
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device)

        max_length = inputs.input_ids[0].size(0) + max_tokens
        if max_length > 2048:
            # dp not even try to generate if the input is too long
            return "too_many_tokens"
        with torch.no_grad():
            x = len(prefix_suffix_tuples[0][0])
            try:
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            except Exception as e:
                if type(e) == IndexError:
                    logger.exception(
                        f"Error in generating code snippet from SantaCoder with an IndexError.",
                    )
                    return "too_many_tokens"
                else:
                    logger.exception(
                        f"Error in generating code snippet from SantaCoder {e}"
                    )
                outputs = None
        try:
            if outputs != None:
                result = [
                    self.extract_fim_part(
                        self.tokenizer.decode(tensor, skip_special_tokens=False)
                    )
                    for tensor in outputs
                ]
                logger.debug(
                    f"SantaCoder Invoked - input = ( {prefix_suffix_tuples} ) - output = {result}"
                )
                return result if output_list else result[0]
            else:
                return None
        except Exception as e:
            logger.exception(f"Error in generating code snippet")


class SantaCoderBlock(InfillModel):
    def __init__(self):
        checkpoint = "bigcode/santacoder"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                revision="comments",
                trust_remote_code=True,
                max_length=200,
            ).to(self.device)
            logger.info(f"SantaCoderBlock model successfuly loaded")
        except Exception as e:
            logger.exception(f"Error in loading the SantaCoderBlock model")

    def predict(self, input_text: str, suffix_text: str) -> str:
        """
        Generate code snippet from the input text

        Args:
            input_text (str): input code. not tokenized.

        Raises:
            e: any error in generating code snippet

        Returns:
            str: geenrated code snippet
        """

        try:
            # inputs: torch.Tensor = self.tokenizer(
            #     input_text, return_tensors="pt", padding=True, return_token_type_ids=False
            # ).to(self.device)
            # max_length = inputs.input_ids[0].size(0) + max_tokens
            # if max_length > 2048:
            #     # dp not even try to generate if the input is too long
            #     return "too_many_tokens"
            inputs: torch.Tensor = self.tokenizer.encode(
                input_text, return_tensors="pt"
            ).to(self.device)
            suffixes = self.tokenizer.encode(suffix_text, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs: torch.Tensor = self.model.generate(
                    inputs,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=suffixes.shape[1],
                )

            logger.debug(
                f"SantaCoderBlock Invoked - input = ( {input_text} ) - output = ( {self.tokenizer.decode(outputs[0])} )"
            )
            return self.tokenizer.decode(outputs[0])
        except Exception as e:
            logger.exception(f"Error in generating code snippet from SantaCoderBlock")
            raise e

    def infill(self, input_text) -> None:
        pass


if __name__ == "__main__":
    test_mdel = SantaCoderBlock()
    out = test_mdel.predict("def test():")
    print(out)
