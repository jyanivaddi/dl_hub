import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def _prepare_batch(self, idx):
        # get a src, target pair
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_str_len = len(enc_input_tokens)
        dec_str_len = len(dec_input_tokens)

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # we will add <s> and </s>
        # we will only add only the <s> token to the decoder
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure that the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long!")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim = 0,
        )

        # Add only the <s>
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_str_lenth": enc_str_len,
            "decoder_str_length": dec_str_len
        }

    def __getitem__(self, idx):
        # get a src, target pair
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_str_len = len(enc_input_tokens)
        dec_str_len = len(dec_input_tokens)

        #print("inside get item and I am returning the dict list!")
        return {
            "encoder_input_tokens": enc_input_tokens,
            "decoder_input_tokens": dec_input_tokens,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_str_length": enc_str_len,
            "decoder_str_length": dec_str_len,
        }

    def collate_samples(self, batch):
        """
        Perform dynamic batching on the sequences.
        For each batch, we get the length of the longest sentence and pad the remaining sentences according to that.
        """

        #print("inside collate function")
        # max encoder str length
        encoder_input_max = max(len(x["encoder_input_tokens"]) for x in batch)
        #print(f"longest encoder input in this batch: {encoder_input_max}")
        decoder_input_max = max(len(x["decoder_input_tokens"]) for x in batch)
        #print(f"longest decoder input in this batch: {decoder_input_max}")

        encoder_inputs = []
        decoder_inputs = []
        encoder_masks = []
        decoder_masks = []
        labels = []
        src_texts = []
        tgt_texts = []

        for cnt, x in enumerate(batch):
            # Add sos, eos and padding to each sentence
            enc_num_padding_tokens = max(0, encoder_input_max - len(x["encoder_input_tokens"]))  # we will add <s> and </s>
            # we will only add only the <s> token to the decoder
            dec_num_padding_tokens = max(0, decoder_input_max - len(x["decoder_input_tokens"]))

            # Add <s> and </s> token
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(x["encoder_input_tokens"], dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

            # Add only the <s>
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(x["decoder_input_tokens"], dtype=torch.int64),
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

            # Add only </s> token
            label = torch.cat(
                [
                    torch.tensor(x["decoder_input_tokens"], dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

            encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
            decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)) # (1, seq_len) & (1, seq_len, seq_len)
            #print(f"{cnt+1}: encoder_inputsize: {len(encoder_input)} encoder_mask_size: {encoder_mask.shape}")
            #print(f"{cnt+1}: decoder_inputsize: {len(decoder_input)} decoder_mask_size: {decoder_mask.shape}")

            # Double check the size of the tensors to make sure they are all seq_len long
            assert encoder_input.size(0) == encoder_input_max + 2  # add SOS and EOS
            assert decoder_input.size(0) == decoder_input_max + 1  # add SOS
            assert label.size(0) == decoder_input_max + 1  # add EOS
            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            labels.append(label)
            encoder_masks.append(encoder_mask.unsqueeze(0)) # we need to do this to preserve the dimensions when doing vstack
            decoder_masks.append(decoder_mask.unsqueeze(0)) # we need to do this to preserve the dimensions when doing vstack
            src_texts.append(x["src_text"])
            tgt_texts.append(x["tgt_text"])

        #print("inside get item and I am returning the dict list!")
        return {
            "encoder_input": torch.vstack(encoder_inputs),
            "decoder_input": torch.vstack(decoder_inputs),
            "encoder_mask": torch.vstack(encoder_masks),
            "decoder_mask": torch.vstack(decoder_masks),
            "label": torch.vstack(labels),
            "src_text": src_texts,
            "tgt_text": tgt_texts
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)),diagonal=1).type(torch.int)
    return mask == 0


