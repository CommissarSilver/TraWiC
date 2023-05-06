import torch, inspect, logger_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logger_utils.CustomLogger(log_file="model.log")


class SantaCoder:
    """
    interface for interacting with the SantaCoder model
    """

    def __init__(self):
        frame = inspect.currentframe()
        frame_info = inspect.getframeinfo(frame)
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
                trust_remote_code=True,
                max_length=200,
            ).to(self.device)
            logger.info(
                f"{frame_info.filename} - {frame_info.function} - SantaCoder model successfuly loaded"
            )
        except Exception as e:
            logger.exception(
                f"{frame_info.filename} - {frame_info.function} - Error in loading the SantaCoder model"
            )
            raise e

    def predict(self, input_text: str):
        frame = inspect.currentframe()
        frame_info = inspect.getframeinfo(frame)
        try:
            logger.info(
                f"{frame_info.filename} - {frame_info.function} - SantaCoder Invoked - input_text = {input_text}"
            )
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model.generate(inputs)
            logger.info(
                f"{frame_info.filename} - {frame_info.function} - SantaCoder Generated Code Snippet - output = {self.tokenizer.decode(outputs[0])}"
            )
            return self.tokenizer.decode(outputs[0])
        except Exception as e:
            logger.exception(
                f"{frame_info.filename} - {frame_info.function} - Error in generating code snippet",
            )
            raise e

    def extract_fim_part(self, s: str):
        """
        Find the index of <fim-middle>
        """
        start = s.find(self.FIM_MIDDLE) + len(self.FIM_MIDDLE)
        stop = s.find(self.ENDOFTEXT, start) or len(s)
        return s[start:stop]

    def infill(
        self,
        prefix_suffix_tuples,
        max_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ):
        output_list = True
        if type(prefix_suffix_tuples) == tuple:
            prefix_suffix_tuples = [prefix_suffix_tuples]
            output_list = False

        prompts = [
            f"{self.FIM_PREFIX}{prefix}{self.FIM_SUFFIX}{suffix}{self.FIM_MIDDLE}"
            for prefix, suffix in prefix_suffix_tuples
        ]
        # `return_token_type_ids=False` is essential, or we get nonsense output.
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device)
        max_length = inputs.input_ids[0].size(0) + max_tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # WARNING: cannot use skip_special_tokens, because it blows away the FIM special tokens.
        result = [
            self.extract_fim_part(
                self.tokenizer.decode(tensor, skip_special_tokens=False)
            )
            for tensor in outputs
        ]
        return result if output_list else result[0]
