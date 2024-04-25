from transformers import AutoTokenizer, RobertaForMaskedLM
import torch
from typing import Tuple


class Roberta:
    def __init__(self):
        checkpoint = "FacebookAI/roberta-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
            self.model = RobertaForMaskedLM.from_pretrained(
                "FacebookAI/roberta-base"
            ).to(self.device)

        except Exception as e:
            print(e)

    def infill(
        self,
        prefix_suffix_tuples: Tuple[str, str, str, str],
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        if len(prefix_suffix_tuples) > 512:
            return "too_many_tokens"
        inputs = self.tokenizer(
            prefix_suffix_tuples, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        mask_token_indices = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(
            as_tuple=False
        )

        predicted_token_ids = [
            logits[i, index].argmax(axis=-1) for i, index in mask_token_indices
        ]
        outputs = predicted_tokens = [
            self.tokenizer.decode(token_id) for token_id in predicted_token_ids
        ]
        return outputs


if __name__ == "__main__":
    roberta = Roberta()
    print(
        roberta.infill(
            [
                "def <mask>: print('hello world')",
                "def print_hello_world: print(<mask>)",
                "def print_name(name): print('hello' + <mask>)",
            ]
        )
    )
