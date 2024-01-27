from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

def get_absolute_paths(directory_path, max_files = None):
    absolute_paths = []

    # Check if the given path is a valid directory
    if os.path.isdir(directory_path):
        # Iterate over all files in the directory
        for root, _, files in tqdm(os.walk(directory_path)):
            for file in files:
                # Construct the absolute path for each file
                absolute_path = os.path.abspath(os.path.join(root, file))
                absolute_paths.append(absolute_path)
                if max_files is not None and len(absolute_paths) > max_files:
                    break
    return absolute_paths

class PickleDataset(Dataset):

    def __init__(self, 
                 ds_path, 
                 tokenizer, 
                 image_embeddings_key = 'clip_embeddings', 
                 caption_key = 'blip_caption',
                 max_embd_len=2048):
        super().__init__()
        self.ds_path = ds_path
        self.tokenizer = tokenizer
        self.ds = None
        self.image_ids = None
        self.image_embeddings_key = image_embeddings_key
        self.caption_key = caption_key
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.max_len_of_sentence = max_embd_len-2

        with open(ds_path, 'r', encoding='UTF-8') as fh:
            self.ds = read_json_file(ds_path)
            self.image_ids = list(self.ds.keys()) 


    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):

        # get image embeddings
        img_embds = self.ds[self.image_ids[idx]][self.image_embeddings_key]
        
        # get caption
        caption = self.ds[self.image_ids[idx]][self.caption_key]
        caption_encoded = self.tokenizer(caption, return_tensors="pt", return_attention_mask=False)
        num_padding_tokens_input = self.max_len_of_sentence - (1+512+len(caption_encoded)+1)
        num_padding_tokens_output = self.max_len_of_sentence - (1+512+len(caption_encoded))

            
        # Add <s> and </s> token
        x = torch.cat(
            [
                self.bos_token,
                img_embds,
                caption_encoded,
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens_input, dtype=torch.int64),
            ],
            dim=0,)

        # Add only the <s>
        y = torch.cat(
            [
                self.bos_token*len(img_embds),
                self.bos_token,
                caption_encoded,
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens_output, dtype=torch.int64),
            ],
            dim=0,
        )

        return {
            "x": x,
            "y": y,
            "input_sentences": input_text,
        }

    def collate_samples(self, batch):
        """
        Perform dynamic batching on the sequences.
        For each batch, we get the length of the longest sentence and pad the remaining sentences according to that.
        """

        #print("inside collate function")
        # max encoder str length
        max_len = max(x["token_len"] for x in batch)
        #print(f"longest encoder input in this batch: {encoder_input_max}")

        x_list = []
        y_list = []
        input_sentences = []

        for cnt, x in enumerate(batch):
            # Add sos, eos and padding to each sentence
            num_padding_tokens_input = max(0, max_len - len(x["input_tokens"]))  # we will add <s> and </s>
            # we will only add only the <s> token to the decoder
            num_padding_tokens_output = num_padding_tokens_input+1

            # Add <s> and </s> token
            batch_x = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(x["input_tokens"], dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * num_padding_tokens_input, dtype=torch.int64),
                ],
                dim=0,
            )

            # Add only the <s>
            batch_y = torch.cat(
                [
                    torch.tensor(x["input_tokens"], dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * num_padding_tokens_output, dtype=torch.int64),
                ],
                dim=0,
            )
            x_list.append(batch_x)
            y_list.append(batch_y)
            input_sentences.append(x["input_sentence"])

        #print("inside get item and I am returning the dict list!")
        return {
            "x": torch.vstack(x_list),
            "y": torch.vstack(y_list),
            "input_sentences": input_sentences,
        }

