# A place for riva to work
This is a repository with information about how to work with riva because their own stuff never works, and has little to no information on the internet

## Converting NeMo models to Riva models (ASR)
Make sure to install dependencies (I find easier to use COLAB because it already has some dependencies installed)
```
# Install dependencies
!pip install nvidia-pyindex
!pip install git+https://github.com/nvidia-riva/nemo2riva.git/ --no-deps
!pip install nvidia-eff --no-deps ipython ipykernel tornado prompt-toolkit  pyzmq
!pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com


## Install NeMo
BRANCH = 'main'
!python -m pip -q install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]

```
You can know call the function that converts models (some NeMo models might notbe available in Riva yet)
```
!nemo2riva --out {riva_model_filepath} --key=nemotoriva {nemo_model_filepath}
```

## Converting Riva models to rmir (ASR)
Essentials
- Riva Service Maker Docker image (riva_sm_image)
- The folder your model is in (model_path)
- Your model file name (model_name)

Run the following docker command (Remember to have [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and configured and to do docker login with an ngc api key)
```
docker run --rm --gpus 0 -v $(pwd)/$model_path:/data #riva_sm_image -- \
    riva-build speech_recognition \
        /data/rmir/asr_offline_finetuned-ctc.rmir:$KEY \
        /data/$model_name.riva:$KEY \
        --offline \
        --name=asr_offline_parakeet_pipeline \
        --decoder_type=greedy \
        --ms_per_timestep=40 \
        --chunk_size=4.8 \
        --left_padding_size=1.6 \
        --right_padding_size=1.6 \
        --max_batch_size=16 \
        --nn.fp16_needs_obey_precision_pass \
        --featurizer.use_utterance_norm_params=False \
        --featurizer.precalc_norm_time_steps=0 \
        --featurizer.precalc_norm_params=False \
        --featurizer.max_batch_size=512 \
        --featurizer.max_execution_batch_size=512 \
        --language_code=en-US
```

## Actually using own models on riva
