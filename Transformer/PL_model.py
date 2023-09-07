import torch
from pytorch_lightning import LightningModule
from .model import build_transformer
#from model import build_transformer

class LitTransformer(LightningModule):
    """
    Pytorch Lightning module for Transformer

    """
    def __init__(self,
                 config,
                 loss_criterion,
                 tokenizer_src,
                 tokenizer_tgt,
                 num_validation_examples=10,
                 epochs=10):
        super().__init__()

        self.config = config
        self.loss_criterion = loss_criterion
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.num_validation_examples = num_validation_examples
        self.epochs = epochs
        self._vocab_src_len = tokenizer_src.get_vocab_size()
        self._vocab_tgt_len = tokenizer_tgt.get_vocab_size()
        self.model = self._define_transformer_model()
        self.scheduler = None
        self.scheduler_dict = {}
        self.optimizer = None
        self.this_step_train_loss = None
        self.predicted_list = []
        self.expected_list = []
        self.save_hyperparameters(ignore=['loss_criterion', 'epoch'])

    def _define_transformer_model(self):
        model = build_transformer(self._vocab_src_len,
                                  self._vocab_tgt_len,
                                  self.config['seq_len'],
                                  self.config['seq_len'],
                                  d_model=self.config['d_model'],
                                  d_ff=self.config['d_ff'],
                                  parameter_sharing=self.config['parameter_sharing'])
        return model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler_dict(self, scheduler, freq='step'):
        self.scheduler = scheduler
        self.scheduler_dict = {
            "scheduler": self.scheduler,
            "interval": freq,
        }

    def configure_optimizers(self):
        if self.scheduler_dict:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler_dict}
        return {"optimizer": self.optimizer}

    def forward(self, x):
        # Run the tensors through the encoder, decoder, and projection layer
        encoder_input = x['encoder_input']  # (B, seq_len)
        decoder_input = x['decoder_input']  # (B, seq_len)
        encoder_mask = x['encoder_mask']  # (B, 1, 1, seq_len)
        decoder_mask = x['decoder_mask']  # (B, 1, seq_len, seq_len)
        encoder_output = self.model.encode(encoder_input,
                                           encoder_mask)  # (B, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output,
                                           encoder_mask,
                                           decoder_input,
                                           decoder_mask)  # (B, seq_len, d_model)
        proj_output = self.model.project(decoder_output)  # (B, seq_len, vocab_size)
        return proj_output


    @staticmethod
    def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

    def greedy_decode(self, encoder_input, encoder_mask):
        sos_idx = self.tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = self.tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it or every step
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input)
        while True:
            if decoder_input.size(1) == self.config['seq_len']:
                break

            # build mask for target
            decoder_mask = LitTransformer.causal_mask(decoder_input.size(1)).type_as(encoder_mask)

            # calculate output
            out = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            # out = out.squeeze(1)

            # get next token
            prob = self.model.project(out[:, -1])
            # prob = model.project(out[-1,:])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item())], dim=1
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)

    def evaluate(self, batch, stage=None):
        """
        Evaluate the model on validation dataset.
        """
        encoder_input = batch['encoder_input']  # (b, seq_len)
        encoder_mask = batch['encoder_mask']  # (b, 1, 1, seq_len)

        # check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        model_out = self.greedy_decode(encoder_input, encoder_mask)

        source_text = batch['src_text'][0]
        target_text = batch['tgt_text'][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())


        if stage:
            # print the source, target, and the model output
            print("*****************************************")
            print(f"{f'SOURCE: ' :>12}{source_text}")
            print(f"{f'TARGET: ' :>12}{target_text}")
            print(f"{f'PREDICTED: ' :>12}{model_out_text}")
            print("*****************************************\n")
        return model_out_text, target_text

    def training_step(self, batch):
        label = batch['label']  # (B, seq_len)
        #proj_output = self(encoder_input, decoder_input, encoder_mask, decoder_mask)
        proj_output = self(batch)
        loss = self.loss_criterion(proj_output.view(-1, self._vocab_tgt_len),
                                   label.view(-1))
        self.log("train_loss", loss.item(), prog_bar=True)
        self.this_step_train_loss = loss.item()
        #self.train_loss(proj_output.view(-1, self._vocab_tgt_len), label.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx < self.num_validation_examples:
            predicted, expected = self.evaluate(batch, "val")
            self.predicted_list.append(predicted)
            self.expected_list.append(expected)


    def test_step(self, batch, batch_idx):
        if batch_idx < self.num_validation_examples:
            self.evaluate(batch, "test")



