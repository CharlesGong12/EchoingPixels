import os
import sys
sys.path.append(os.getcwd())

from swift.llm import (
    InferRequest, Model, ModelGroup, ModelInfo, ModelMeta, PtEngine, RequestConfig, TemplateMeta,
    TemplateType,
    register_model, register_template
    )
from swift.llm.model.model.qwen import patch_qwen_vl_utils, patch_get_input_embeddings
from swift.llm.model.register import get_model_tokenizer_with_flash_attn
from swift.llm.model.utils import use_submodel_func
from swift.llm.model.model_arch import ModelArch

from swift.llm.template.template.qwen import Qwen2_5OmniTemplate, QwenTemplateMeta

from swift.utils import get_env_args


def get_model_tokenizer_qwen2_5_omni_compress(model_dir, *args, **kwargs):
    from qwen2_5_omni_compress.modeling_qwen2_5_omni_compress import Qwen2_5OmniCompressForConditionalGeneration
    from qwen2_5_omni_compress.configuration_qwen2_5_omni_compress import Qwen2_5OmniCompressConfig
    from transformers import Qwen2_5OmniProcessor
    from qwen_omni_utils import vision_process
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2_5OmniCompressForConditionalGeneration
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = Qwen2_5OmniCompressConfig.from_pretrained(model_dir, trust_remote_code=True)
    patch_qwen_vl_utils(vision_process)
    kwargs['model_config'].enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, False)
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    copy_init_bidir = os.getenv("COPY_INIT_BIDIR", "0") in ("1", "true", "True")
    if copy_init_bidir:
        model.thinker.model.init_bidir_from_decoder()
    if model:
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        use_submodel_func(base_model, 'thinker')
        base_model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        base_model.config.talker_config.pad_token_id = None
        patch_get_input_embeddings(base_model.thinker.visual, 'patch_embed')
    return model, processor

register_template(QwenTemplateMeta('qwen2_5_omni_compress', template_cls=Qwen2_5OmniTemplate))

register_model(
    ModelMeta(
        'qwen2_5_omni_compress',
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-Omni-3B', 'Qwen/Qwen2.5-Omni-3B'),
                Model('Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen2.5-Omni-7B'),
            ]),
        ],
        'qwen2_5_omni_compress',
        get_model_tokenizer_qwen2_5_omni_compress,
        model_arch=ModelArch.qwen2_5_omni,
        architectures=['Qwen2_5OmniModel', 'Qwen2_5OmniCompressForConditionalGeneration'],
        requires=['transformers>=4.50', 'soundfile', 'qwen_omni_utils', 'decord'],
        tags=['vision', 'video', 'audio'],
        additional_saved_files=['spk_dict.pt'],
        ignore_patterns=[],
        is_multimodal=True,
    ))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['USE_AUDIO_IN_VIDEO'] = 'True'
    os.environ['VIDEO_MAX_PIXELS'] = '100356'
    os.environ['FORCE_QWENVL_VIDEO_READER'] = 'torchvision'
    os.environ['ENABLE_AUDIO_OUTPUT'] = 'False'

    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    from qwen2_5_omni_compress.modeling_qwen2_5_omni_compress import Qwen2_5OmniCompressForConditionalGeneration
    from qwen2_5_omni_compress.configuration_qwen2_5_omni_compress import Qwen2_5OmniCompressConfig

    AutoConfig.register("qwen2_5_omni_compress", Qwen2_5OmniCompressConfig)
    AutoModel.register(Qwen2_5OmniCompressConfig, Qwen2_5OmniCompressForConditionalGeneration)
    AutoModelForCausalLM.register(Qwen2_5OmniCompressConfig, Qwen2_5OmniCompressForConditionalGeneration)

    # NOTE: the asset paths below are placeholders — provide your own local or
    # publicly available image/video files before running.
    infer_request = [
        InferRequest(messages=[{'role': 'user', 'content': 'Describe this image'}], images=['assets/example.jpg']),
        InferRequest(messages=[{'role': 'user', 'content': 'Describe this video'}], videos=['assets/example.mp4']),
        InferRequest(messages=[{'role': 'user', 'content': 'who are you'}]),
    ]
    request_config = RequestConfig(max_tokens=512, temperature=0)

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} available GPU(s).")
    for i in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(device=i)

    # Replace with the path to your compressed checkpoint.
    MODEL_PATH = '/path/to/Qwen2.5-Omni-Compress-3B'
    engine = PtEngine(MODEL_PATH, model_type='qwen2_5_omni_compress', device_map='auto', attn_impl="sdpa", torch_dtype=torch.bfloat16,)

    print("\n--- Memory allocated after model load ---")
    total_allocated_after_load = 0
    for i in range(num_gpus):
        # memory_allocated = current state (not peak) — we care about the steady state here
        mem_allocated = torch.cuda.memory_allocated(device=i) / 1024**2 # MB
        total_allocated_after_load += mem_allocated
        print(f"GPU {i}: allocated {mem_allocated:.2f} MB")
    print(f"Total allocated: {total_allocated_after_load:.2f} MB")
    print("--------------------------\n")

    # Reset peak stats again to measure the *inference* overhead in isolation.
    print("Reset peak stats, starting inference...")
    for i in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(device=i)

    response = engine.infer(infer_request, request_config)

    print("\n--- Inference peak memory report ---")
    total_peak_allocated = 0
    for i in range(num_gpus):
        peak_allocated = torch.cuda.max_memory_allocated(device=i) / 1024**2 # MB
        total_peak_allocated += peak_allocated
        print(f"GPU {i} peak allocated: {peak_allocated:.2f} MB")

    print("---")
    print(f"Total peak allocated across GPUs: {total_peak_allocated:.2f} MB")
    print("--------------------------\n")

    print(f'raw response: {response}')
    swift_response = response[0].choices[0].message.content

    print(f'response: {swift_response}')
