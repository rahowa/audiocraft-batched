from pathlib import Path
import omegaconf
import torch
import typing as tp
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenProcessor

from transformers.models.musicgen.modeling_musicgen import shift_tokens_right

import bz2
import pickle
from datetime import datetime
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from audiocraft.solvers.builders import DatasetType, get_audio_datasets
# from audiocraft.utils.cache import CachedBatchLoader, CachedBatchWriter
from accelerate import Accelerator, cpu_offload


cfg = omegaconf.OmegaConf.merge(
    omegaconf.OmegaConf.load("config/solver/musicgen/default.yaml"),
    omegaconf.OmegaConf.load("config/solver/default.yaml"),
    omegaconf.OmegaConf.load("config/model/lm/default.yaml"),
    omegaconf.OmegaConf.load("config/solver/compression/default.yaml"),
    omegaconf.OmegaConf.load("config/solver/compression/encodec_musicgen_32khz.yaml"),
    omegaconf.OmegaConf.load("config/solver/musicgen/musicgen_melody_32khz.yaml"),
    omegaconf.OmegaConf.load("config/model/lm/musicgen_lm.yaml"),
    omegaconf.OmegaConf.load("config/config.yaml"),
    omegaconf.OmegaConf.load("config/conditioner/chroma2music.yaml"),
    omegaconf.OmegaConf.load("config/dset/internal/music_phonk_32khz.yaml")   
)
cfg.dataset.segment_duration = 30

cache_path = "./cache_dir"
cache_path = None
use_cached_writer = False
use_cached_reader = False

cached_batch_writer = None
cached_batch_loader = None
cfg_dropout_p = 0.1
attribute_dropout_p = {"default": 0.1}
current_stage = "train"
device = "cpu"
cfg.device = device

               
def collate_fn_wrapper(collate_fn, processor):
    stop_words = [
        'BLADE',
        'Crystals',
        'DJ',
        'DVRST',
        'Dxrk',
        'GLICHERY',
        'GUDOG',
        'Hensonn',
        'INTERWORLD',
        'KIIXSHI',
        'KORDHELL',
        'MANO',
        'Memory',
        'NEON',
        'Onimxru',
        'Reboot',
        'SCXR',
        'XDXVIL',
        'g3ox_em',
        'yatashigang'
    ]
    def wrapper(samples):
        batch = collate_fn(samples)
        wavs, infos = batch
        texts_with_new_word = []
        texts_without_word = []
        for info in infos:
            text = info.description or ""
            text_orig = text.lower() 
            for stop_word in stop_words:
                text_orig = text_orig.replace(stop_word.lower(), "")
            text_modified = text.lower().replace("phonk", "")
            texts_with_new_word.append(text_orig)
            texts_without_word.append(text_modified)
        without_word_data = processor(text=texts_without_word, padding=True, return_tensors="pt")
        with_word_data =  processor(
            text=texts_with_new_word,
            audio=[wav[0].numpy() for wav in wavs],
            sampling_rate=32_000, #infos[0].self_wav.sample_rate[0],
            padding=True,
            # max_length=100,
            return_tensors="pt"
        )
        with_word_data.update({
            "input_ids_class": without_word_data["input_ids"],
            "encoder_attention_mask": with_word_data["attention_mask"],
            "encoder_attention_mask_cls": without_word_data["attention_mask"],
            })
        return with_word_data
    return wrapper

import numpy as np 


def downcast_arr(arr):
    if arr.dtype == np.int64:
        return arr.astype(np.int32)
    elif arr.dtype == np.float64:
        return arr.astype(np.float32)
    else:
        return arr


def build_saving_fn(savedir: tp.Union[str, Path]) -> tp.Callable[[dict[str, torch.Tensor]], None]:
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True)
    counter = 0
    def save_batch_data(batch: dict[str, torch.Tensor]) -> None:
        idx =  0
        nonlocal counter
        while True:
            generated_filename = f"sample-{counter}-{idx}-{datetime.now()}.npz"
            try:
                # with bz2.BZ2File(savedir/generated_filename, "wb") as rf:
                #     pickle.dump({k: v[idx].cpu().numpy() for k, v  in batch.items()}, rf)
                np.savez(str(savedir/generated_filename), **{k: downcast_arr(v[idx].cpu().numpy()) for k, v in batch.items()})
                idx+=1
            except IndexError:
                # (savedir/generated_filename).unlink()
                break
        counter+=1
    return save_batch_data
    
    
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


if __name__ == "__main__":
    batch_size = 6
    accelerator = Accelerator(mixed_precision="fp16", cpu=False, device_placement=True)
    processor: MusicgenProcessor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    full_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    pattern_provider = DelayedPatternProvider(4, delays=[0, 1, 2, 3])
    num_codebooks = full_model.decoder.num_codebooks
    config = full_model.config
    generation_config = full_model.generation_config
    
    full_model.config.device = accelerator.device
    full_model.config.decoder_start_token_id = full_model.generation_config.decoder_start_token_id
    full_model.config.pad_token_id = full_model.generation_config.pad_token_id

    text_encoder = full_model.text_encoder
    audio_encoder = full_model.audio_encoder
    decoder = full_model.decoder
    cfg.dataset.batch_size = batch_size
    dataloader = get_audio_datasets(cfg, DatasetType.MUSIC)["train"]
    audio_encoder.eval()
    text_encoder.eval()
    decoder.eval()

    enc_to_dec_proj = full_model.enc_to_dec_proj
    enc_to_dec_proj.eval()
    enc_to_dec_proj.requires_grad_(False)
    # text_encoder.to(accelerator.device)#, dtype=torch.float16)
    dataloader.collate_fn = collate_fn_wrapper(dataloader.collate_fn, processor)
    dataloader.num_workers = 8
    dataloader.shuffle = True
    enc_to_dec_proj, text_encoder, audio_encoder, decoder, dataloader = accelerator.prepare(
        enc_to_dec_proj, text_encoder, audio_encoder, decoder, dataloader
    )
    with torch.inference_mode():
        for epoch in range(1):
            saving_fn = build_saving_fn("generated_train_dataset")
            print(f"{len(dataloader)=}")
            for idx, batch in enumerate(dataloader):
                for k, v in batch.items():
                    if v.dtype == torch.float32:
                        batch[k] = v.half()
                if idx >= 3000:
                    break
                
                audio_encoder_outputs = audio_encoder(
                    input_values=batch["input_values"],
                    padding_mask=batch["padding_mask"]
                    # **kwargs_audio_encoder,
                )
                audio_codes = audio_encoder_outputs.audio_codes
                frames, bsz, codebooks, seq_len = audio_codes.shape
                if frames != 1:
                    raise ValueError(
                        f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                        "disabled by setting `chunk_length=None` in the audio encoder."
                    )
                labels = audio_codes[0, ...].reshape(batch_size * num_codebooks, seq_len)
                    
                input_ids = batch["input_ids"]
                input_ids_cls = batch["input_ids_class"]
                encoder_attention_mask = batch["encoder_attention_mask"]
                encoder_attention_mask_cls = batch["encoder_attention_mask_cls"]
                
                encoder_outputs_cls = text_encoder(
                    input_ids=input_ids_cls,
                    attention_mask=encoder_attention_mask_cls,
                    output_hidden_states=True,
                    return_dict=config.use_return_dict,
                )
                encoder_outputs = text_encoder(
                    input_ids=input_ids,
                    attention_mask=encoder_attention_mask,
                    output_hidden_states=True,
                    return_dict=config.use_return_dict,
                )
                encoder_hidden_states = encoder_outputs[0]
                encoder_hidden_states_cls = encoder_outputs_cls[0]

                if (
                    text_encoder.config.hidden_size != full_model.decoder.config.hidden_size
                    and full_model.decoder.config.cross_attention_hidden_size is None
                ):
                    encoder_hidden_states = enc_to_dec_proj(encoder_hidden_states)
                    encoder_hidden_states_cls = enc_to_dec_proj(encoder_hidden_states_cls)

                if encoder_attention_mask is not None:
                    encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[..., None] #.to(accelerator.device)
                    encoder_hidden_states_cls = encoder_hidden_states_cls * encoder_attention_mask_cls[..., None] #.to(accelerator.device)
                decoder_input_ids = shift_tokens_right(labels, config.pad_token_id, config.decoder_start_token_id)
                result_prepared_data_batch_cls = {
                    "input_ids": decoder_input_ids.reshape(batch_size, num_codebooks, seq_len),
                    "encoder_attention_mask": encoder_attention_mask_cls,
                    "encoder_hidden_states": encoder_hidden_states_cls,
                    "labels": labels.reshape(batch_size, num_codebooks, seq_len)
                }
                logits_cls, logits_mask_cls = compute_predictions(decoder, generation_config, pattern_provider, result_prepared_data_batch_cls)
                result_prepared_data_batch = {
                    "input_ids": decoder_input_ids.reshape(batch_size, num_codebooks, seq_len),
                    "encoder_attention_mask": encoder_attention_mask,
                    "encoder_attention_mask_cls": encoder_attention_mask_cls,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_hidden_states_cls": encoder_hidden_states_cls,
                    "labels": labels.reshape(batch_size, num_codebooks, seq_len), 
                    "logits_cls": logits_cls,
                    # "logits_mask_cls": logits_mask_cls,
                    "padding_mask": batch["padding_mask"],
                }
                saving_fn(result_prepared_data_batch)
