# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, logging, inspect

logging.basicConfig(
    filename="model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SantaCoder:
    def __init__(self):
        frame = inspect.currentframe()
        frame_info = inspect.getframeinfo(frame)

        checkpoint = "bigcode/santacoder"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint, trust_remote_code=True
            ).to(self.device)
            logging.info(
                f"{frame_info.filename} - {frame_info.function} - SantaCoder model successfuly loaded"
            )
        except Exception as e:
            logging.exception(
                f"{frame_info.filename} - {frame_info.function} - Error in loading the SantaCoder model"
            )
            raise e

    def predict(self, input_text):
        frame = inspect.currentframe()
        frame_info = inspect.getframeinfo(frame)
        try:
            logging.info(
                f"{frame_info.filename} - {frame_info.function} - SantaCoder Invoked - input_text = {input_text}"
            )
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(
                self.device
            )
            outputs = self.model.generate(inputs)
            logging.info(
                f"{frame_info.filename} - {frame_info.function} - SantaCoder Generated Code Snippet - output = {self.tokenizer.decode(outputs[0])}"
            )
            return self.tokenizer.decode(outputs[0])
        except Exception as e:
            logging.exception(
                f"{frame_info.filename} - {frame_info.function} - Error in generating code snippet",
            )
            raise e
