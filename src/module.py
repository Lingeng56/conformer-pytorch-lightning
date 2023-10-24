import torch
import pytorch_lightning as pl
import os
import sentencepiece as spm
from scheduler import WarmupLR
from torchmetrics import WordErrorRate


class TransducerModule(pl.LightningModule):

    def __init__(self,
                 model,
                 ckpt_path,
                 char_dict,
                 bpe_model,
                 warmup_steps=25000,
                 lr=0.01,
                 streaming_eval=False,
                 decoding_chunk_size=0,
                 num_decoding_left_chunks=-1
                 ):
        super(TransducerModule, self).__init__()
        # Define Model
        self.model = model

        # Define Attributions
        self.warmup_steps = warmup_steps
        self.lr = lr

        # For Checkpoint
        self.ckpt_path = ckpt_path

        # For predict
        self.out_stream = open('tmp_prediction.txt', 'w')
        self.char_dict = char_dict
        self.streaming_eval = streaming_eval
        self.decoding_chunk_size = decoding_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

        # For metric
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model)
        self.train_wer = WordErrorRate()
        self.valid_wer = WordErrorRate()
        self.validation_preds = []
        self.validation_truth = []


    def training_step(self, batch, batch_idx):
        sorted_keys, padded_feats, feats_length, padded_labels, label_lengths, transcripts = batch
        loss_dict = self.model(batch)
        loss_rnnt = loss_dict['loss_rnnt']
        # loss_attn = loss_dict['loss_attn']
        loss_ctc = loss_dict['loss_ctc']
        loss = loss_dict['loss']
        curr_lr = self.lr_schedulers().get_last_lr()[0]


        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True,
                 batch_size=padded_feats.size(0))
        self.log('train_ctc_loss', loss_ctc, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True,
                 batch_size=padded_feats.size(0))
        # self.log('train_attn_loss', loss_attn, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True,
        #          batch_size=padded_feats.size(0))
        self.log('train_rnnt_loss', loss_rnnt, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True,
                 batch_size=padded_feats.size(0))
        self.log('lr', curr_lr, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True,
                 batch_size=padded_feats.size(0))
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sorted_keys, padded_feats, feats_length, padded_labels, label_lengths, transcripts = batch
        preds = self.predict_step(batch, batch_idx)
        self.valid_wer.update(preds, transcripts)
        self.validation_truth += transcripts
        self.validation_preds += preds
        self.log('valid_wer', self.valid_wer.compute(), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True,
                 batch_size=padded_feats.size(0))

    def on_validation_epoch_start(self):
        if self.local_rank == 0:
            self.out_stream = open('tmp_prediction.txt', 'w')

    def on_validation_epoch_end(self):
        if self.local_rank == 0:
            self.validation_preds = self.all_gather(self.validation_preds)
            self.validation_truth = self.all_gather(self.validation_truth)
            print('Saving checkpoint to %s' % self.ckpt_path)
            path = os.path.join(self.ckpt_path, f'Step:{self.global_step}-Valid_WER:{self.valid_wer(self.validation_preds, self.validation_truth):.6f}.ckpt')
            self.trainer.save_checkpoint(path)
            self.trainer.save_checkpoint(os.path.join(self.ckpt_path, 'last.ckpt'))
            self.out_stream.close()
        self.validation_preds = []
        self.validation_truth = []
        self.trainer.strategy.barrier()



    def on_predict_epoch_start(self):
        if self.local_rank == 0:
            self.out_stream = open('tmp_prediction.txt', 'w')
        self.trainer.strategy.barrier()

    def on_predict_epoch_end(self):
        if self.local_rank == 0:
            self.out_stream.close()
        self.trainer.strategy.barrier()


    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sorted_keys, padded_feats, feats_length, padded_labels, label_lengths, transcripts = batch
        preds = []
        for key, feat, feat_length, transcript in zip(sorted_keys, padded_feats, feats_length, transcripts):
            feat = feat.unsqueeze(0)
            feat_length = feat_length.unsqueeze(0)
            if self.streaming_eval:
                hyps = self.model.greedy_search_streaming_eval(
                    feat,
                    self.decoding_chunk_size,
                    self.num_decoding_left_chunks
                )
            else:
                hyps = self.model.greedy_search(feat,
                                                feat_length)

            content = []
            for w in hyps:
                if w == self.model.eos:
                    break
                content.append(self.char_dict[w])
            text = f'Key: {key}\nPred: {self.sp.decode(content)}\nTruth: {transcript}'
            self.out_stream.write(text + '\n')
            preds.append(self.sp.decode(content))
        return preds



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = WarmupLR(optimizer, warmup_steps=self.warmup_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
