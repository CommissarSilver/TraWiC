# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from models.model import InfillModel
from peft import LoraConfig, PeftModel
import logging
from typing import Tuple

logger = logging.getLogger("model")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

input_text = "<fim-prefix>def fib(n):<fim-suffix>    else:\n        return fib(n - 2) + fib(n - 1)<fim-middle>"
inputs = tokenizer.encode(
    input_text,
    return_tensors="pt",
    return_token_type_ids=False,
).to(device)
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))


class MistralCoder(InfillModel):
    def __init__(self) -> None:
        # toknizer config
        self.FIM_PREFIX = "<fim-prefix>"
        self.FIM_MIDDLE = "<fim-middle>"
        self.FIM_SUFFIX = "<fim-suffix>"
        self.FIM_PAD = "<fim-pad>"
        self.ENDOFTEXT = "<|endoftext|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/vamaj/scratch/TraWiC/llms/mistral",
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self.FIM_PREFIX,
                    self.FIM_MIDDLE,
                    self.FIM_SUFFIX,
                    self.FIM_PAD,
                ],
            }
        )
        # model config
        self.device_map = {"": 0}
        base_model_path = "/home/vamaj/scratch/TraWiC/llms/mistral"
        adapter_path = "/home/vamaj/scratch/TraWiC/llms/mistral_fim"
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=self.device_map,
                local_files_only=True,
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
            )
            self.model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Mistral-FIM model successfuly loaded")
        except Exception as e:
            logger.exception(f"Error in loading the Mistral-FIM model")
            raise Exception("Problem in initializing Mistral-FIM Model")

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
                f"Mistral-FIM Invoked - input = ( {input_text} ) - output = ( {self.tokenizer.decode(outputs[0])} )"
            )
            return self.tokenizer.decode(outputs[0])
        except Exception as e:
            logger.exception(f"Error in generating code snippet from Mistral-FIM")
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
            logger.exception(f"Error in extracting fim part from Mistral-FIM output")
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
                        f"Error in generating code snippet from Mistral-FIM with an IndexError.",
                    )
                    return "too_many_tokens"
                else:
                    logger.exception(
                        f"Error in generating code snippet from Mistral-FIM {e}"
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
                    f"Mistral-FIM Invoked - input = ( {prefix_suffix_tuples} ) - output = {result}"
                )
                return result if output_list else result[0]
            else:
                return None
        except Exception as e:
            logger.exception(f"Error in generating code snippet")