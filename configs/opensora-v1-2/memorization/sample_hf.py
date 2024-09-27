# First stage training 
# Webvid-10M datasets (40k hours) 30k steps (2 epochs).
# video length 2s~16s. We use the original caption in the dataset for training. 
# The training config locates in stage1.py.
resolution = "360p" # training resolution: 240p and 360p 
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    force_huggingface=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
    force_huggingface=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None

# Condition
prompt_path = "./assets/webvid-10M-mini.txt"
prompt = None  # prompt has higher priority than prompt_path

# Others
batch_size = 8
# seed = 42
save_dir = "./samples/samples-webvid-10M-mini-bf16/"