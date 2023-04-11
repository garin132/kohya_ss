Validating that requirements are satisfied.
All requirements satisfied.
Load CSS...
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Folder 150_wpnr : 1500 steps
max_train_steps = 1500
stop_text_encoder_training = 0
lr_warmup_steps = 150
accelerate launch --num_cpu_threads_per_process=2 "train_db.py" --enable_bucket --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --train_data_dir="C:/Users/mycar/Downloads/NR/image" --resolution=512,512 --output_dir="C:/Users/mycar/Downloads/NR/model" --logging_dir="C:/Users/mycar/Downloads/NR/log" --save_model_as=safetensors --output_name="tellme" --max_data_loader_n_workers="0" --learning_rate="1e-5" --lr_scheduler="cosine" --lr_warmup_steps="150" --train_batch_size="1" --max_train_steps="1500" --save_every_n_epochs="1" --mixed_precision="fp16" --save_precision="fp16" --cache_latents --optimizer_type="AdamW" --max_data_loader_n_workers="0" --bucket_reso_steps=64 --mem_eff_attn --xformers --bucket_no_upscale
prepare tokenizer
prepare images.
found directory C:\Users\mycar\Downloads\NR\image\150_wpnr contains 10 image files
1500 train images with repeating.
0 reg images.
no regularization images / 正則化画像が見つかりませんでした
[Dataset 0]
  batch_size: 1
  resolution: (512, 512)
  enable_bucket: True
  min_bucket_reso: 256
  max_bucket_reso: 1024
  bucket_reso_steps: 64
  bucket_no_upscale: True

  [Subset 0 of Dataset 0]
    image_dir: "C:\Users\mycar\Downloads\NR\image\150_wpnr"
    image_count: 10
    num_repeats: 150
    shuffle_caption: False
    keep_tokens: 0
    caption_dropout_rate: 0.0
    caption_dropout_every_n_epoches: 0
    caption_tag_dropout_rate: 0.0
    color_aug: False
    flip_aug: False
    face_crop_aug_range: None
    random_crop: False
    token_warmup_min: 1,
    token_warmup_step: 0,
    is_reg: False
    class_tokens: wpnr
    caption_extension: .caption


[Dataset 0]
loading image sizes.
100%|█████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 598.25it/s]
make buckets
min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます
number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）
bucket 0: resolution (512, 512), count: 1500
mean ar error (without repeats): 0.0
prepare accelerator
Using accelerator 0.15.0 or above.
load Diffusers pretrained models
safety_checker\model.safetensors not found
Fetching 19 files: 100%|███████████████████████████████████████████████████████████████████████| 19/19 [00:00<?, ?it/s]
C:\Windows\System32\kohya_ss\venv\lib\site-packages\transformers\models\clip\feature_extraction_clip.py:28: FutureWarning: The class CLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use CLIPImageProcessor instead.
  warnings.warn(
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
Replace CrossAttention.forward to use FlashAttention (not xformers)
[Dataset 0]
caching latents.
 50%|█████████████████████████████████████████▌                                         | 5/10 [00:15<00:10,  2.12s/it]Loading config...
Loading config...
100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [00:26<00:00,  2.66s/it]
prepare optimizer, data loader etc.
use AdamW optimizer | {}
Traceback (most recent call last):
  File "C:\Windows\System32\kohya_ss\train_db.py", line 427, in <module>
    train(args)
  File "C:\Windows\System32\kohya_ss\train_db.py", line 191, in train
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\accelerate\accelerator.py", line 876, in prepare
    result = tuple(
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\accelerate\accelerator.py", line 877, in <genexpr>
    self._prepare_one(obj, first_pass=True, device_placement=d) for obj, d in zip(args, device_placement)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\accelerate\accelerator.py", line 741, in _prepare_one
    return self.prepare_model(obj, device_placement=device_placement)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\accelerate\accelerator.py", line 912, in prepare_model
    model = model.to(self.device)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\torch\nn\modules\module.py", line 927, in to
    return self._apply(convert)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\torch\nn\modules\module.py", line 579, in _apply
    module._apply(fn)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\torch\nn\modules\module.py", line 579, in _apply
    module._apply(fn)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\torch\nn\modules\module.py", line 579, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\torch\nn\modules\module.py", line 602, in _apply
    param_applied = fn(param)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\torch\nn\modules\module.py", line 925, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
RuntimeError: CUDA out of memory. Tried to allocate 16.00 MiB (GPU 0; 4.00 GiB total capacity; 2.66 GiB already allocated; 0 bytes free; 2.71 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Traceback (most recent call last):
  File "C:\Users\mycar\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Users\mycar\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "C:\Windows\System32\kohya_ss\venv\Scripts\accelerate.exe\__main__.py", line 7, in <module>
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\accelerate\commands\accelerate_cli.py", line 45, in main
    args.func(args)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\accelerate\commands\launch.py", line 1104, in launch_command
    simple_launcher(args)
  File "C:\Windows\System32\kohya_ss\venv\lib\site-packages\accelerate\commands\launch.py", line 567, in simple_launcher
    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
subprocess.CalledProcessError: Command '['C:\\Windows\\System32\\kohya_ss\\venv\\Scripts\\python.exe', 'train_db.py', '--enable_bucket', '--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5', '--train_data_dir=C:/Users/mycar/Downloads/NR/image', '--resolution=512,512', '--output_dir=C:/Users/mycar/Downloads/NR/model', '--logging_dir=C:/Users/mycar/Downloads/NR/log', '--save_model_as=safetensors', '--output_name=tellme', '--max_data_loader_n_workers=0', '--learning_rate=1e-5', '--lr_scheduler=cosine', '--lr_warmup_steps=150', '--train_batch_size=1', '--max_train_steps=1500', '--save_every_n_epochs=1', '--mixed_precision=fp16', '--save_precision=fp16', '--cache_latents', '--optimizer_type=AdamW', '--max_data_loader_n_workers=0', '--bucket_reso_steps=64', '--mem_eff_attn', '--xformers', '--bucket_no_upscale']' returned non-zero exit status 1.
