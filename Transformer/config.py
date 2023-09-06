from pathlib import Path

def get_config():
    return {
        "parameter_sharing": False,
        "d_ff": 2048,
        "batch_size": 32,
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 160,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": True,
        "tokenizer_file": "tokenizer_{0}.json",
        "rundir": "runs",
        "experiment_name": "tmodel_dynamic_pad",
        "ds_mode": "Online",
        #"ds_path": "/home/e183534/OpusBooks",
        "ds_path": None,
        "ds_name": "opus_books",
        "save_ds_to_disk": True
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)