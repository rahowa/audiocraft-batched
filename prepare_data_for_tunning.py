from pathlib import Path
import omegaconf
import torch
import typing as tp
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenProcessor

from transformers.models.musicgen.modeling_musicgen import shift_tokens_right

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
    def wrapper(samples):
        batch = collate_fn(samples)
        wavs, infos = batch
        return processor(
            text=[info.description or "" for info in infos],
            audio=[wav[0].numpy() for wav in wavs],
            sampling_rate=32_000, #infos[0].self_wav.sample_rate[0],
            padding=True,
            return_tensors="pt"
        )
    return wrapper

def build_saving_fn(savedir: tp.Union[str, Path]) -> tp.Callable[[dict[str, torch.Tensor]], None]:
    import bz2
    import pickle
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True)
    counter = 0
    def save_batch_data(batch: dict[str, torch.Tensor]) -> None:
        idx =  0
        nonlocal counter
        while True:
            generated_filename = f"sample-{counter}-{idx}.bz"
            try:
                with bz2.BZ2File(savedir/generated_filename, "wb") as rf:
                    pickle.dump({k: v[idx].cpu().numpy() for k, v  in batch.items()}, rf)
                idx+=1
            except IndexError:
                (savedir/generated_filename).unlink()
                break
        counter+=1
    return save_batch_data
    
    

if __name__ == "__main__":
    
    accelerator = Accelerator(mixed_precision="fp16", cpu=False, device_placement=True)
    processor: MusicgenProcessor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    full_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    num_codebooks = full_model.decoder.num_codebooks
    config = full_model.config
    generation_config = full_model.generation_config
    
    full_model.config.device = accelerator.device
    full_model.config.decoder_start_token_id = full_model.generation_config.decoder_start_token_id
    full_model.config.pad_token_id = full_model.generation_config.pad_token_id

    text_encoder = full_model.text_encoder
    audio_encoder = full_model.audio_encoder
    cfg.dataset.batch_size = 6
    dataloader = get_audio_datasets(cfg, DatasetType.MUSIC)["train"]
    audio_encoder.eval()
    text_encoder.eval()
    
    enc_to_dec_proj = full_model.enc_to_dec_proj
    enc_to_dec_proj.eval()
    enc_to_dec_proj.requires_grad_(False)
    # text_encoder.to(accelerator.device)#, dtype=torch.float16)
    dataloader.collate_fn = collate_fn_wrapper(dataloader.collate_fn, processor)
    enc_to_dec_proj, text_encoder, audio_encoder, dataloader = accelerator.prepare(
        enc_to_dec_proj, text_encoder, audio_encoder, dataloader
    )
    with torch.inference_mode():
        for epoch in range(1):
            saving_fn = build_saving_fn("generated_train_dataset")
            print(f"{len(dataloader)=}")
            for idx, batch in enumerate(dataloader):
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
                labels = audio_codes[0, ...].reshape(-1, num_codebooks, seq_len)
                    
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"] 
                
                encoder_outputs = text_encoder(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True,
                    return_dict=config.use_return_dict,
                )
                encoder_hidden_states = encoder_outputs[0]
                if (
                    text_encoder.config.hidden_size != full_model.decoder.config.hidden_size
                    and full_model.decoder.config.cross_attention_hidden_size is None
                ):
                    encoder_hidden_states = enc_to_dec_proj(encoder_hidden_states)

                if attention_mask is not None:
                    encoder_hidden_states = encoder_hidden_states * attention_mask[..., None] #.to(accelerator.device)
                decoder_input_ids = shift_tokens_right(labels, config.pad_token_id, config.decoder_start_token_id)

                result_prepared_data_batch = {
                    "decoder_input_ids": decoder_input_ids,
                    "attention_mask": attention_mask,
                    "encoder_hidden_states": encoder_hidden_states,
                    "labels": labels, 
                    "padding_mask": batch["padding_mask"],
                }
                for k, v in result_prepared_data_batch.items():
                    print(k, v.size())
                saving_fn(result_prepared_data_batch)