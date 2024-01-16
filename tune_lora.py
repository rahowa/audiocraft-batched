from functools import partial
from typing import List
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
from transformers import MusicgenForCausalLM
import omegaconf
import torch
import torch.nn.functional as F
import typing as tp
from transformers import MusicgenForConditionalGeneration

from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from accelerate import Accelerator


cfg = omegaconf.OmegaConf.merge(
    omegaconf.OmegaConf.load("config/solver/musicgen/default.yaml"),
    omegaconf.OmegaConf.load("config/solver/default.yaml"),
    omegaconf.OmegaConf.load("config/model/lm/default.yaml"),
    omegaconf.OmegaConf.load("config/solver/compression/default.yaml"),
    omegaconf.OmegaConf.load("config/solver/compression/encodec_musicgen_32khz.yaml"),
    omegaconf.OmegaConf.load("config/solver/musicgen/musicgen_base_32khz.yaml"),
    omegaconf.OmegaConf.load("config/model/lm/musicgen_lm.yaml"),
    omegaconf.OmegaConf.load("config/config.yaml"),
    omegaconf.OmegaConf.load("config/conditioner/chroma2music.yaml"),
    omegaconf.OmegaConf.load("config/dset/audio/example.yaml")   
)
cfg.dataset.segment_duration = 30

cache_path = "./cache_dir"
cache_path = None

cfg_dropout_p = 0.1 
attribute_dropout_p = {"default": 0.1}
current_stage = "train"
device = "cpu"
cfg.device = device


def compute_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
    """Compute cross entropy between multi-codebook targets and model's logits.
    The cross entropy is computed per codebook to provide codebook-level cross entropy.
    Valid timesteps for each of the codebook are pulled from the mask, where invalid
    timesteps are set to 0.

    Args:
        logits (torch.Tensor): Model's logits of shape [B, K, T, card].
        targets (torch.Tensor): Target codes, of shape [B, K, T].
        mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
    Returns:
        ce (torch.Tensor): Cross entropy averaged over the codebooks
        ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
    """
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    ce = torch.zeros([], device=targets.device)
    ce_per_codebook: tp.List[torch.Tensor] = []
    for k in range(K):
        logits_k = (
            logits[:, k, ...].contiguous().view(-1, logits.size(-1))
        )  # [B x T, card]
        targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
        mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
        ce_targets = targets_k[mask_k]
        ce_logits = logits_k[mask_k]
        q_ce = F.cross_entropy(ce_logits, ce_targets)
        ce += q_ce
        ce_per_codebook.append(q_ce.detach())
    # average cross entropy across codebooks
    ce = ce / K
    return ce, ce_per_codebook


class PreparedDataset(Dataset):
    def  __init__(self, path) -> None:
        super().__init__()
        self.path = Path(path) 
        self.files = list(self.path.iterdir())
        
    def __len__(self) -> int:
        return len(self.files)
    
    def read_file(self, filename: tp.Union[str, Path]) -> dict[str, torch.Tensor]:
        try:
            # with bz2.BZ2File(filename, "rb") as compressed_data_file:
            #     data = pickle.load(compressed_data_file)
            # return data
            return np.load(filename)
        except:
            print(filename)
        
    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        values = dict(self.read_file(self.files[index]))
        # print(values.keys())
        return values

    @staticmethod
    def pad_sequences(samples: List[np.ndarray], pad_value: int) -> List[np.ndarray]:
        max_len = max(sample.shape[0] for sample in samples)
        for idx, sample in enumerate(samples):
            res = np.zeros((max_len, *sample.shape[1:]), dtype=sample.dtype) + pad_value
            res[:sample.shape[0]] = sample 
            samples[idx] = res 
        return samples 
    
    @staticmethod
    def collate_fn(samples: List[dict[str, torch.Tensor]], use_cls: bool = False) -> dict[str, torch.Tensor]:
        
        for idx, att_mask in enumerate(PreparedDataset.pad_sequences([s["encoder_attention_mask"] for s in samples], 0)):
            samples[idx]["encoder_attention_mask"] = att_mask

        for idx, ehs in enumerate(PreparedDataset.pad_sequences([s["encoder_hidden_states"] for s in samples], 0)):
            samples[idx]["encoder_hidden_states"] = ehs

        if use_cls:
            for idx, att_mask in enumerate(PreparedDataset.pad_sequences([s["encoder_attention_mask_cls"] for s in samples], 0)):
                samples[idx]["encoder_attention_mask_cls"] = att_mask
                
            for idx, ehs in enumerate(PreparedDataset.pad_sequences([s["encoder_hidden_states_cls"] for s in samples], 0)):
                samples[idx]["encoder_hidden_states_cls"] = ehs

        res = defaultdict(list)
        for sample in samples:
            for k, v in sample.items():
                res[k].append(v)
        return {k: torch.from_numpy(np.array(v)) for k, v in res.items()}
        
            
def compute_predictions(decoder, generation_config, pattern_provider, batch):
    codes = batch["labels"]
    B, K, T = codes.shape
    codes = codes.contiguous()
    # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
    pattern = pattern_provider.get_pattern(T)
    sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
        codes, generation_config.decoder_start_token_id, keep_only_valid_steps=True
    )
    # apply model on pattern sequence
    logits = decoder(
        input_ids=sequence_codes,
        encoder_hidden_states=batch["encoder_hidden_states"],
        encoder_attention_mask=batch["encoder_attention_mask"]
    ).logits
    logits = logits.view(B, K, logits.size(1), logits.size(2))
    # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
    # and provide the corresponding mask over invalid positions of tokens
    logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
    # note: we use nans as special token to make it obvious if we feed unexpected logits
    logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
        logits, float('nan'), keep_only_valid_steps=True
    )
    logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
    logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
    return logits, logits_mask


def main():
    accelerator = Accelerator(mixed_precision="fp16")
    full_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cpu")
    decoder: MusicgenForCausalLM = full_model.decoder.to(accelerator.device)
    decoder.train()
    full_config = full_model.config
    generation_config = full_model.generation_config
    del full_model
    config = decoder.config
    config.device = accelerator.device
    config.decoder_start_token_id = full_config.decoder_start_token_id
    config.pad_token_id = generation_config.pad_token_id
    pattern_provider = DelayedPatternProvider(4, delays=[0, 1, 2, 3])
    modules_to_lora = [
        name
        for name, _ in decoder.named_modules()
        if name.endswith("q_proj") or name.endswith("v_proj")
    ] + ["lm_heads.0","lm_heads.1", "lm_heads.2", "lm_heads.3"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=modules_to_lora,
    )
    batch_size = 1
    epochs = 2
    use_cls_loss = True
    decoder = get_peft_model(decoder, peft_config)
    dataset = PreparedDataset("generated_train_dataset")
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8, collate_fn=partial(dataset.collate_fn, use_cls=use_cls_loss))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=0.01,  betas=(0.9, 0.95))
    loss_fct = torch.nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(len(dataloader) * epochs * 0.1), num_training_steps=int(len(dataloader) * epochs)
    )
    decoder, optimizer, dataloader, scheduler = accelerator.prepare(decoder, optimizer, dataloader, scheduler)
    for epoch in range(epochs):
        decoder.train()
        for batch in dataloader:
            with accelerator.accumulate(decoder):
                print("start decoding...")
                for k, v in batch.items():
                    if v.dtype == torch.float:
                        batch[k] = v.half()
                logits, logits_mask = compute_predictions(decoder, generation_config, pattern_provider, batch)
                mask = torch.ones_like(batch["labels"]) & logits_mask
                loss, _ = compute_cross_entropy(logits, batch["labels"].long(), mask)
                if use_cls_loss:
                    decoder_outputs_cls = dict(
                        labels=batch["labels"],
                        encoder_hidden_states=batch["encoder_hidden_states_cls"],
                        encoder_attention_mask=batch["encoder_attention_mask_cls"]   
                    )
                    logits_cls, _ = compute_predictions(decoder, generation_config, pattern_provider, decoder_outputs_cls)
                    loss_cls = loss_fct(logits_cls.view(-1, config.vocab_size), batch["logits_cls"].view(-1, config.vocab_size).softmax(dim=1))
                    print(f"Loss cls: {loss_cls}")
                    loss = loss + loss_cls
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print(f"Loss: {loss}")
        decoder.eval()
        accelerator.unwrap_model(decoder).save_pretrained(f"tuned-musicgen-small-lora-{epoch}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        decoder.eval()
        accelerator.unwrap_model(decoder).save_pretrained("tuned-musicgen-small-lora")
    accelerator.end_training()


if __name__ == "__main__":
    main()
