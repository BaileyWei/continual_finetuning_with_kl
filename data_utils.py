
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class DataCollatorWithPaddingForGLM:
    r"""
    DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None



    def tokenize_batch_element(
        self,
        prompt: str,
        answer: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """

        max_seq_length = self.max_length
        self.sptoken = self.tokenizer.encode(text="")[-2:]

        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
        ds = TokenTruncation.process(self.tokenizer, self.tokenizer.eos_token_id, a_ids, b_ids, max_seq_length, self.sptoken,)

        return ds[0]

    def collate(self, batch):
        """
        reference: chatglm_finetuning from Ssbuild
        https://github.com/ssbuild/chatglm_finetuning
        """

        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])


        max_len = torch.max(o.pop('seqlen')).tolist()
        b_input_ids = o['input_ids'][:, :max_len]
        ctxlens = o.pop('ctxlen')
        if ctxlens is None:
            ctxlens = [None] * len(b_input_ids)

        b_position_ids,b_attention_mask = [],[]
        for input_ids,context_length in zip(b_input_ids,ctxlens):
            context_length = context_length.squeeze(dim=-1)
            mask_position = context_length - 1
            position_ids = list(range(context_length)) + [mask_position] * (max_len - context_length)
            block_position_ids = [0] * context_length + list(range(1, max_len - context_length + 1))


            attention_mask = torch.ones((1, max_len, max_len))
            attention_mask = torch.tril(attention_mask)
            attention_mask[..., :context_length] = 1
            attention_mask = (attention_mask < 0.5)

            b_position_ids.append(torch.stack((torch.tensor(position_ids),torch.tensor(block_position_ids))))
            b_attention_mask.append(attention_mask)

        b_attention_mask = torch.stack(b_attention_mask, dim=0)
        b_position_ids = torch.stack(b_position_ids,dim=0)

        o['input_ids'] = b_input_ids.long()
        o['attention_mask'] = b_attention_mask.bool()
        o['position_ids'] = b_position_ids.long()
        o['labels'] = o['labels'][:, :max_len].long()

        return o

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            answer = feature["answer"]

            batch_element = self.tokenize_batch_element(prompt, answer)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)


class TokenTruncation:
    """
    reference: chatglm_finetuning from Ssbuild
    https://github.com/ssbuild/chatglm_finetuning
    """
    @classmethod
    def process(cls, tokenizer,eos_token_id, a_ids, b_ids, max_seq_length, sptoken, ensure_answer_min_length=1):
        ds = []

        assert ensure_answer_min_length > 0
        input_ids_qa = a_ids[:max_seq_length-len(sptoken)-ensure_answer_min_length] + sptoken + b_ids + [eos_token_id] * 2
        pos = 0
        while pos < len(input_ids_qa):
            if sptoken[0] in input_ids_qa[pos:pos + max_seq_length]:
                val = input_ids_qa[pos:pos + max_seq_length][-1]
                if val == sptoken[-1]:
                    input_ids = input_ids_qa[pos+1:pos + max_seq_length+1]
                    pos += max_seq_length + 1
                elif val == sptoken[0]:
                    input_ids = input_ids_qa[pos + 2:pos + max_seq_length + 2]
                    pos += max_seq_length + 2
                else:
                    input_ids = input_ids_qa[pos:pos + max_seq_length]
                    pos += max_seq_length
            else:
                input_ids = sptoken + input_ids_qa[pos:pos + max_seq_length - 2]
                pos += max_seq_length - 2

            d = TokenIdsFinal.process(input_ids,sptoken,max_seq_length,tokenizer)
            ds.append(d)
        return ds


class TokenIdsFinal:
    """
    reference: chatglm_finetuning from Ssbuild
    https://github.com/ssbuild/chatglm_finetuning
    """
    @classmethod
    def process(cls,input_ids,sptoken,max_seq_length,tokenizer):
        ctxlen = input_ids.index(sptoken[-1])
        mask_position = ctxlen - 1
        labels = [-100] * ctxlen + input_ids[mask_position + 1:]

        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        ctxlen = np.asarray(ctxlen, dtype=np.int32)
        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))

        d = {
            'input_ids': input_ids,
            'labels': labels,
            'seqlen': seqlen,
            'ctxlen': ctxlen
        }
        return d
