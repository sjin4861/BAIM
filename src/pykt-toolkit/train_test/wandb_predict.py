import os
import argparse
import json
import copy
import glob
import shutil
import sys
import torch
import pandas as pd
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pykt.models import evaluate, evaluate_question, load_model
from pykt.datasets import init_test_datasets

try:
    from pykt.config import que_type_models
except ImportError:
    que_type_models = [
        "dkt",
        "dkt+",
        "dkt_forget",
        "kqn",
        "sakt",
        "saint",
        "atkt",
        "atktfix",
        "gkt",
        "skvmn",
        "hawkes",
        "qdkt",
        "qikt",
    ]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
try:
    with open("../configs/wandb.json") as fin:
        wandb_config = json.load(fin)
except FileNotFoundError:
    wandb_config = {"api_key": ""}


def ensure_expected_ckpt(save_dir, emb_type):
    expected_ckpt = os.path.join(save_dir, f"{emb_type}_model.ckpt")
    if os.path.isfile(expected_ckpt):
        return

    ckpt_candidates = sorted(glob.glob(os.path.join(save_dir, "*.ckpt")))
    if not ckpt_candidates:
        return

    preferred = [
        path
        for path in ckpt_candidates
        if emb_type in os.path.basename(path) or os.path.basename(path).endswith("_model.ckpt")
    ]
    source_ckpt = preferred[0] if preferred else ckpt_candidates[0]

    try:
        os.symlink(os.path.basename(source_ckpt), expected_ckpt)
        print(
            f"[ckpt-fallback] Linked {os.path.basename(source_ckpt)} -> {os.path.basename(expected_ckpt)}"
        )
    except OSError:
        shutil.copy2(source_ckpt, expected_ckpt)
        print(
            f"[ckpt-fallback] Copied {os.path.basename(source_ckpt)} -> {os.path.basename(expected_ckpt)}"
        )


def save_detailed_predictions(model, test_loader, save_path, model_name):
    print(f"Extracting detailed predictions to {save_path}...")
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    elif hasattr(model, "eval"):
        try:
            model.eval()
        except TypeError:
            for module in model.modules():
                module.training = False
    results = []
    with torch.no_grad():
        for data in test_loader:
            if model_name in ["dkt_forget", "datakt"]:
                dcur, dgaps = data
            else:
                dcur = data
            if isinstance(dcur, dict):
                for key in dcur:
                    if isinstance(dcur[key], torch.Tensor):
                        dcur[key] = dcur[key].to(device)
            if model_name in que_type_models and model_name not in ["lpkt", "promptkt"]:
                y = model.predict_one_step(dcur)
            else:
                y = model(dcur)
            qshft = dcur["shft_qseqs"]
            rshft = dcur["shft_rseqs"]
            sm = dcur["smasks"]
            mask = sm.bool()
            valid_qs = torch.masked_select(qshft, mask).cpu().numpy()
            valid_rs = torch.masked_select(rshft, mask).cpu().numpy()
            valid_ys = torch.masked_select(y, mask).cpu().numpy()
            for q, r, p in zip(valid_qs, valid_rs, valid_ys):
                results.append(
                    {"question_id": int(q), "true_label": int(r), "pred_prob": float(p)}
                )
    if results:
        df_pred = pd.DataFrame(results)
        df_pred.to_csv(save_path, index=False)
        print(f"Extraction Done! Saved {len(df_pred)} rows.")
    else:
        print("Warning: No results extracted.")


def predict_for_one_dir(save_dir, batch_size, fusion_type, params):

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
            if remove_item in model_config:
                del model_config[remove_item]
        if "emb_path" in model_config:
            del model_config["emb_path"]

        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name = trained_params["model_name"]
        dataset_name = trained_params["dataset_name"]
        emb_type = trained_params["emb_type"]

        if model_name in [
            "saint",
            "saint++",
            "sakt",
            "atdkt",
            "simplekt",
            "bakt_time",
            "sakt_que",
            "saint_que",
        ]:
            train_config = config["train_config"]
            model_config["seq_len"] = train_config["seq_len"]
    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]

    skip_window_eval = params.get("skip_window_eval", 0) == 1
    if model_name in que_type_models:
        window_key = "test_window_file_quelevel"
        base_key = "test_file_quelevel"
    else:
        window_key = "test_window_file"
        base_key = "test_file"

    window_path = None
    if window_key in data_config:
        window_path = os.path.join(data_config["dpath"], data_config[window_key])

    if window_path is not None and not os.path.exists(window_path):
        print(
            f"[window-fallback] Missing window file: {window_path}. Disable window evaluation."
        )
        skip_window_eval = True
        if base_key in data_config:
            # Keep dataset initialization alive even when window split is absent.
            data_config[window_key] = data_config[base_key]

    if "emb_path" in trained_params:
        data_config["emb_path"] = trained_params["emb_path"]
    if model_name not in ["dimkt"]:
        (
            test_loader,
            test_window_loader,
            test_question_loader,
            test_question_window_loader,
        ) = init_test_datasets(data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        (
            test_loader,
            test_window_loader,
            test_question_loader,
            test_question_window_loader,
        ) = init_test_datasets(
            data_config, model_name, batch_size, diff_level=diff_level
        )

    print(
        f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}"
    )

    ensure_expected_ckpt(save_dir, emb_type)
    model = load_model(model_name, model_config, data_config, emb_type, save_dir)
    detailed_save_path = os.path.join(save_dir, "detailed_test_predictions.csv")
    save_detailed_predictions(model, test_loader, detailed_save_path, model_name)
    save_test_path = ""
    if params["save_all_preds"] == 1:
        save_test_path = os.path.join(
            save_dir, model.emb_type + "_test_predictions.txt"
        )
        print(f"Predictions will be saved to {save_test_path}")

    if model.model_name == "rkt":
        testauc, testavg_prc, testacc = evaluate(
            model, test_loader, model_name, None, save_path=save_test_path
        )
    else:
        testauc, testavg_prc, testacc = evaluate(
            model, test_loader, model_name, save_path=save_test_path
        )
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = ""
    if params["save_all_preds"] == 1:
        save_test_window_path = os.path.join(
            save_dir, model.emb_type + "_test_window_predictions.txt"
        )

    if skip_window_eval:
        print("Skip window evaluation.")
    else:
        if model.model_name == "rkt":
            window_testauc, window_testavg_prc, window_testacc = evaluate(
                model,
                test_window_loader,
                model_name,
                None,
                save_path=save_test_window_path,
            )
        else:
            window_testauc, window_testavg_prc, window_testacc = evaluate(
                model, test_window_loader, model_name, save_path=save_test_window_path
            )

        print(f"window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    dres = {
        "testauc": testauc,
        "testacc": testacc,
        "window_testauc": window_testauc,
        "window_testacc": window_testacc,
    }

    q_testaucs, q_testaccs = -1, -1
    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(
            save_dir, model.emb_type + "_test_question_predictions.txt"
        )
        q_testaucs, q_testaccs = evaluate_question(
            model,
            test_question_loader,
            model_name,
            fusion_type,
            save_path=save_test_question_path,
        )
        for key in q_testaucs:
            dres["oriauc" + key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc" + key] = q_testaccs[key]

    print(dres)

    if params["use_wandb"] == 1:
        try:
            wandb.log(dres)
        except:
            pass

    results_save_path = os.path.join(save_dir, "prediction_results.json")
    with open(results_save_path, "w") as json_file:
        json.dump(dres, json_file, indent=2)

    return dres


def get_fold_dirs(parent_dir):
    child_dirs = []
    for name in sorted(os.listdir(parent_dir)):
        path = os.path.join(parent_dir, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
            child_dirs.append(path)
    return child_dirs


def read_fold_index(save_dir):
    config_path = os.path.join(save_dir, "config.json")
    try:
        with open(config_path) as fin:
            config = json.load(fin)
        return config.get("params", {}).get("fold", -1)
    except Exception:
        return -1


def main(params):
    if params["use_wandb"] == 1:
        import wandb

        if "wandb_project_name" in params and params["wandb_project_name"] != "":
            wandb.init(project=params["wandb_project_name"])
        else:
            wandb.init()

    save_dir = params["save_dir"]
    batch_size = params["bz"]
    fusion_type = params["fusion_type"].split(",")

    single_config_path = os.path.join(save_dir, "config.json")
    if os.path.isfile(single_config_path):
        predict_for_one_dir(save_dir, batch_size, fusion_type, params)
        return

    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"save_dir does not exist: {save_dir}")

    fold_dirs = get_fold_dirs(save_dir)
    if len(fold_dirs) == 0:
        raise FileNotFoundError(
            f"No fold directories with config.json found under: {save_dir}"
        )

    print(f"[aggregate] Found {len(fold_dirs)} fold directories under {save_dir}")
    fold_results = []
    for fold_dir in fold_dirs:
        fold_idx = read_fold_index(fold_dir)
        print(f"[aggregate] Running fold={fold_idx}, dir={fold_dir}")
        dres = predict_for_one_dir(fold_dir, batch_size, fusion_type, params)
        fold_results.append(
            {
                "fold": fold_idx,
                "save_dir": fold_dir,
                "testauc": float(dres.get("testauc", np.nan)),
                "testacc": float(dres.get("testacc", np.nan)),
                "window_testauc": float(dres.get("window_testauc", np.nan)),
                "window_testacc": float(dres.get("window_testacc", np.nan)),
            }
        )

    valid_testaucs = [x["testauc"] for x in fold_results if not np.isnan(x["testauc"])]
    if len(valid_testaucs) == 0:
        print("[aggregate] No valid testauc values found.")
        return

    testauc_mean = float(np.mean(valid_testaucs))
    testauc_std = float(np.std(valid_testaucs))
    print("\n[aggregate] testauc summary")
    print(f"[aggregate] folds={len(valid_testaucs)}")
    print(f"[aggregate] testauc mean+-std: {testauc_mean:.6f} +- {testauc_std:.6f}")

    summary = {
        "num_folds": len(valid_testaucs),
        "testauc_mean": testauc_mean,
        "testauc_std": testauc_std,
        "fold_results": fold_results,
    }
    summary_path = os.path.join(save_dir, "prediction_results_summary.json")
    with open(summary_path, "w") as fout:
        json.dump(summary, fout, indent=2)
    print(f"[aggregate] Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--save_all_preds", type=int, default=0)
    parser.add_argument("--wandb_project_name", type=str, default="")
    parser.add_argument("--skip_window_eval", type=int, default=0)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
