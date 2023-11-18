from ast import List
import bz2
from pathlib import Path
import pickle
from transformers import AutoModelForTextEncoding, AutoModel, MusicgenForCausalLM
import warnings
import omegaconf
import torch
import torch.nn.functional as F
import typing as tp
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenProcessor
from transformers.optimization import AdamW

from transformers.models.musicgen.modeling_musicgen import shift_tokens_right
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from audiocraft.data.info_audio_dataset import AudioInfo
from audiocraft.data.music_dataset import MusicDataset, MusicInfo, Paraphraser, augment_music_info_description
from torch.utils.data import DataLoader
from audiocraft.models.builders import get_conditioner_provider

from audiocraft.modules.conditioners import AttributeDropout, ClassifierFreeGuidanceDropout, ConditioningAttributes, ConditioningProvider, SegmentWithAttributes, WavCondition
from audiocraft.solvers.builders import DatasetType, get_audio_datasets
from audiocraft.solvers.compression import CompressionSolver
# from audiocraft.utils.cache import CachedBatchLoader, CachedBatchWriter
from audiocraft.utils.utils import get_dataset_from_loader
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
    omegaconf.OmegaConf.load("config/dset/audio/example.yaml")   
)
cfg.dataset.segment_duration = 30

cache_path = "./cache_dir"
cache_path = None
use_cached_writer = False
use_cached_reader = False

cached_batch_writer = None
cached_batch_loader = None
# if cache_path is not None and use_cached_writer:
#     cached_batch_writer = CachedBatchWriter(Path(cache_path))
# else:
#     cached_batch_loader = CachedBatchLoader(
#         Path(cache_path), cfg.dataset.batch_size, cfg.dataset.num_workers,
#         min_length=cfg.optim.updates_per_epoch or 1)

cfg_dropout_p = 0.1 
attribute_dropout_p = {"default": 0.1}
# cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout_p)
# att_dropout = AttributeDropout(p=attribute_dropout_p)
current_stage = "train"
device = "cpu"
cfg.device = device

# condition_provider = get_conditioner_provider(cfg["transformer_lm"]["dim"], cfg)
# compression_model = CompressionSolver.wrapped_model_from_checkpoint(cfg, cfg.compression_model_checkpoint, device=device)

def prepare_tokens_and_attributes(
    batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
    check_synchronization_points: bool,
    condition_provider,
    compression_model,
    
) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
    """Prepare input batchs for language model training.

    Args:
        batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
            and corresponding metadata as SegmentWithAttributes (with B items).
        check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
    Returns:
        Condition tensors (dict[str, any]): Preprocessed condition attributes.
        Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
            with B the batch size, K the number of codebooks, T_s the token timesteps.
        Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
    _cached_batch_loader = None
    er = None"""
    if model.training:
        warnings.warn(
            "Up to version 1.0.1, the _prepare_tokens_and_attributes was evaluated with `torch.no_grad()`. "
            "This is inconsistent with how model were trained in the MusicGen paper. We removed the "
            "`torch.no_grad()` in version 1.1.0. Small changes to the final performance are expected. "
            "Really sorry about that."
        )
    if cached_batch_loader is None or current_stage != "train":
        audio, infos = batch
        audio = audio.to(device)
        audio_tokens = None
        assert audio.size(0) == len(infos), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(infos)})",
        )
    else:
        audio = None
        # In that case the batch will be a tuple coming from the _cached_batch_writer bit below.
        (infos,) = batch  # type: ignore
        assert all([isinstance(info, AudioInfo) for info in infos])
        assert all([info.audio_tokens is not None for info in infos])  # type: ignore
        audio_tokens = torch.stack([info.audio_tokens for info in infos]).to(device)  # type: ignore
        audio_tokens = audio_tokens.long()
        for info in infos:
            if isinstance(info, MusicInfo):
                # Careful here, if you want to use this condition_wav (e.b. chroma conditioning),
                # then you must be using the chroma cache! otherwise the code will try
                # to use this segment and fail (by that I mean you will see NaN everywhere).
                info.wav = WavCondition(
                    torch.full([1, info.channels, info.total_frames], float("NaN")),
                    length=torch.tensor([info.n_frames]),
                    sample_rate=[info.sample_rate],
                    path=[info.meta.path],
                    seek_time=[info.seek_time],
                )
                dataset = get_dataset_from_loader(dataloader)
                assert isinstance(dataset, MusicDataset), type(dataset)
                if dataset.paraphraser is not None and info.description is not None:
                    # Hackingly reapplying paraphraser when using cache.
                    info.description = dataset.paraphraser.sample_paraphrase(
                        info.meta.path, info.description
                    )
    # prepare attributes
    attributes = [info.to_condition_attributes() for info in infos]
    print(attributes[0])
    # attributes = cfg_dropout(attributes)
    # attributes = att_dropout(attributes)
    tokenized = condition_provider.tokenize(attributes)
    # Now we should be synchronization free.
    if device == "cuda" and check_synchronization_points:
        torch.cuda.set_sync_debug_mode("warn")

    if audio_tokens is None:
        with torch.no_grad():
            audio_tokens, scale = compression_model.encode(audio)
            assert scale is None, "Scaled compression model not supported with LM."

    # with autocast:
    condition_tensors = condition_provider(tokenized)

    # create a padding mask to hold valid vs invalid positions
    padding_mask = torch.ones_like(
        audio_tokens, dtype=torch.bool, device=audio_tokens.device
    )
    # replace encodec tokens from padded audio with special_token_id
    if cfg.tokens.padding_with_special_token:
        audio_tokens = audio_tokens.clone()
        padding_mask = padding_mask.clone()
        token_sample_rate = compression_model.frame_rate
        B, K, T_s = audio_tokens.shape
        for i in range(B):
            n_samples = infos[i].n_frames
            audio_sample_rate = infos[i].sample_rate
            # take the last token generated from actual audio frames (non-padded audio)
            valid_tokens = math.floor(
                float(n_samples) / audio_sample_rate * token_sample_rate
            )
            audio_tokens[i, :, valid_tokens:] = model.special_token_id
            padding_mask[i, :, valid_tokens:] = 0

    if device == "cuda" and check_synchronization_points:
        torch.cuda.set_sync_debug_mode("default")

    if cached_batch_writer is not None and current_stage == "train":
        assert cached_batch_loader is None
        assert audio_tokens is not None
        for info, one_audio_tokens in zip(infos, audio_tokens):
            assert isinstance(info, AudioInfo)
            if isinstance(info, MusicInfo):
                assert not info.joint_embed, "joint_embed and cache not supported yet."
                info.wav = None
            assert one_audio_tokens.max() < 2**15, one_audio_tokens.max().item()
            info.audio_tokens = one_audio_tokens.short().cpu()
        cached_batch_writer.save(infos)

    return condition_tensors, audio_tokens, padding_mask


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


# def run_step( idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict:
#         """Perform one training or valid step on a given batch."""
#         check_synchronization_points = idx == 1 and device == 'cuda'

#         condition_tensors, audio_tokens, padding_mask = _prepare_tokens_and_attributes(
#             batch, check_synchronization_points)


#         if check_synchronization_points:
#             torch.cuda.set_sync_debug_mode('warn')

#         with autocast:
#             model_output = model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
#             logits = model_output.logits
#             mask = padding_mask & model_output.mask
#             ce, ce_per_codebook = _compute_cross_entropy(logits, audio_tokens, mask)
#             loss = ce
#         deadlock_detect.update('loss')

#         if check_synchronization_points:
#             torch.cuda.set_sync_debug_mode('default')

#         if is_training:
#             metrics['lr'] = optimizer.param_groups[0]['lr']
#             if scaler is not None:
#                 loss = scaler.scale(loss)
#             deadlock_detect.update('scale')
#             if cfg.fsdp.use:
#                 loss.backward()

#             if scaler is not None:
#                 scaler.unscale_(optimizer)
#             if cfg.optim.max_norm:
#                 if cfg.fsdp.use:
#                     metrics['grad_norm'] = model.clip_grad_norm_(cfg.optim.max_norm)  # type: ignore
#                 else:
#                     metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), cfg.optim.max_norm
#                     )
#             if scaler is None:
#                 optimizer.step()
#             else:
#                 scaler.step(optimizer)
#                 scaler.update()
#             if lr_scheduler:
#                 lr_scheduler.step()
#             optimizer.zero_grad()
#             deadlock_detect.update('optim')
#             if scaler is not None:
#                 scale = scaler.get_scale()
#                 metrics['grad_scale'] = scale
#             if not loss.isfinite().all():
#                 raise RuntimeError("Model probably diverged.")

#         metrics['ce'] = ce
#         metrics['ppl'] = torch.exp(ce)
#         for k, ce_q in enumerate(ce_per_codebook):
#             metrics[f'ce_q{k + 1}'] = ce_q
#             metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)

#         return metrics


from torch.utils.data import Dataset

class PreparedDataset(Dataset):
    def  __init__(self, path) -> None:
        super().__init__()
        self.path = Path(path) 
        self.files = list(self.path.iterdir())
        
    def __len__(self) -> int:
        return len(self.files)
    
    def read_file(self, filename: tp.Union[str, Path]) -> dict[str, torch.Tensor]:
        try:
            with bz2.BZ2File(filename, "rb") as compressed_data_file:
                data = pickle.load(compressed_data_file)
            return data
        except:
            print(filename)
        
    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        return self.read_file(self.files[index])
    
    
    # @staticmethod
    # def collate_fn(samples: List[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    #     res = {}
    #     for idx, datadict in enumerate(samples):
            
        
if __name__ == "__main__":
    accelerator = Accelerator(mixed_precision="bf16", cpu=False, device_placement=True)
    decoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").decoder
    config = decoder.config
    generation_config = decoder.generation_config
    config.device = accelerator.device
    config.decoder_start_token_id = generation_config.decoder_start_token_id
    config.pad_token_id = generation_config.pad_token_id
    decoder.eval()

    modules_to_lora = [
        name
        for name, _ in decoder.named_modules()
        if name.endswith("q_proj") or name.endswith("v_proj")
    ]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=modules_to_lora,
    )
    
    decoder = get_peft_model(decoder, peft_config)
    dataset = PreparedDataset("generated_train_dataset")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)
    optimizer = AdamW(decoder.parameters(), lr=3e-6, weight_decay=5e-3)
    loss_fct = torch.nn.CrossEntropyLoss()
    decoder, optimizer, dataloader = accelerator.prepare(decoder, optimizer, dataloader)
    # with torch.inference_mode():
    for epoch in range(10):
        for batch in dataloader:
                
            print("start decoding...")
            # print(batch.keys())
            # exit()
            # decoder.print_trainable_parameters()
            decoder_outputs = decoder(
                input_ids=batch["decoder_input_ids"],
                encoder_hidden_states=batch["encoder_hidden_states"],
                encoder_attention_mask=batch["attention_mask"]
            )
            logits = decoder_outputs.logits if config.use_return_dict else decoder_outputs[0]
            loss = loss_fct(logits.view(-1, config.vocab_size), batch["labels"].view(-1))
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            print(f"Loss: {loss}")
        decoder.save_pretrained("tuned-musicgen-small-lora")
                # res = prepare_tokens_and_attributes(batch, False, condition_provider, compression_model)
                # break
                # batch_sample_idx = 0
                # print(type(batch))
                # wav, info = batch
                # print(type(wav), wav.shape, type(info), len(info), type(info[0]))
                # print(info[0].self_wav.sample_rate[0])
                # print(wav[batch_sample_idx][0])
                # res = processor(
                #     text=["80s pop track with bassy drums and synth"],
                #     audio=wav[batch_sample_idx][0],
                #     sampling_rate=info[batch_sample_idx].self_wav.sample_rate[0],
                #     padding=True,
                #     return_tensors="pt"
                # )
                # print(res)
                # print(res["input_values"].shape)
                # audio_encoder_outputs = model.audio_encoder(
                #     input_values=wav[batch_sample_idx].unsqueeze(0),
                #     padding_mask=res["padding_mask"],
                #     # **kwargs_audio_encoder,
                # )
                # audio_codes = audio_encoder_outputs.audio_codes
                # frames, bsz, codebooks, seq_len = audio_codes.shape
                # if frames != 1:
                #     raise ValueError(
                #         f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                #         "disabled by setting `chunk_length=None` in the audio encoder."
                #     )
                # decoder_input_ids = audio_codes[0, ...].reshape(bsz * model.decoder.num_codebooks, seq_len)
                # # print(len(res))
                # # print(res[0]["description"])
                # # condition_tensors, audio_tokens, padding_mask = res 
                # preds = model(**res, labels=decoder_input_ids).loss
                # print(preds)
                # print(preds.shape)
                # break
            # break
