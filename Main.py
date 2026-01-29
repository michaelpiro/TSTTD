import os

from Train_eval import train, eval, select_best

model_conf = {
    "state": "train",  # train, eval, or select_best
    "epoch": 20,
    "band": 103,  # 189, 189, 162, 205
    "multiplier": 2,
    "seed": 1,
    "batch_size": 64,
    "group_length": 20,
    "depth": 4,
    "heads": 4,
    "dim_head": 64,
    "mlp_dim": 64,
    "adjust": False,
    "channel": 128,
    "lr": 1e-4,
    "epision": 5,
    "grad_clip": 1.,
    "device": "cuda:0",
    "training_load_weight": None,
    "save_dir": "./Checkpoint/",
    "test_load_weight": "ckpt_5_.pt",
    "path": "synthetic_of_test7_t.mat"
}

def main(model_config=None):
    modelConfig = {
        "state": "train",  # train, eval, or select_best
        "epoch": 20,
        "band": 103,  # 189, 189, 162, 205
        "multiplier": 2,
        "seed": 1,
        "batch_size": 64,
        "group_length": 20,
        "depth": 4,
        "heads": 4,
        "dim_head": 64,
        "mlp_dim": 64,
        "adjust": False,
        "channel": 128,
        "lr": 1e-4,
        "epision": 5,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_dir": "./Checkpoint/",
        "test_load_weight": "ckpt_5_.pt",
        "path": "synthetic_of_test7_t.mat"
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "eval":
        eval(modelConfig)
    else:
        select_best(modelConfig)


if __name__ == '__main__':
    main(model_config=model_conf)
    checkpoints_dir = os.path.join(model_conf['save_dir'], model_conf['path'])

    model_conf['path'] = "synthetic_of_test72_t.mat"
    new_checkpoints_dir = os.path.join(model_conf['save_dir'], model_conf['path'])
    # rename the name of checkpoint dir
    os.rename(checkpoints_dir, new_checkpoints_dir)
    model_conf["state"] = "select_best"
    main(model_config=model_conf)


