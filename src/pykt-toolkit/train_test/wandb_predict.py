import os
import argparse
import json
import copy
import torch
import pandas as pd
import numpy as np

from pykt.models import evaluate, evaluate_question, load_model
from pykt.datasets import init_test_datasets
# [추가] 모델 타입 확인을 위한 config import
try:
    from pykt.config import que_type_models
except ImportError:
    # 만약 import가 안 되면 직접 정의 (Fallback)
    que_type_models = ["dkt", "dkt+", "dkt_forget", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes", "qdkt", "qikt"]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

# wandb config 파일 로드 (경로가 안 맞으면 에러 날 수 있으므로 예외처리)
try:
    with open("../configs/wandb.json") as fin:
        wandb_config = json.load(fin)
except FileNotFoundError:
    wandb_config = {"api_key": ""}

# --- [추가된 함수] 상세 예측 결과(CSV) 저장 ---
def save_detailed_predictions(model, test_loader, save_path, model_name):
    print(f"Extracting detailed predictions to {save_path}...")
    
    # 1. [FIX] 모델 Eval 모드 전환 (QueBaseModel 호환성)
    if hasattr(model, 'model') and hasattr(model.model, 'eval'):
        model.model.eval()
    elif hasattr(model, 'eval'):
        try:
            model.eval()
        except TypeError:
            # QueBaseModel 등에서 eval() 인자가 안 맞을 경우 강제 설정
            for module in model.modules():
                module.training = False
    
    results = []
    
    with torch.no_grad():
        for data in test_loader:
            # 2. 데이터 전처리
            if model_name in ["dkt_forget", "datakt"]:
                dcur, dgaps = data
            else:
                dcur = data
            
            # 데이터를 device로 이동
            if isinstance(dcur, dict):
                for key in dcur:
                    if isinstance(dcur[key], torch.Tensor):
                        dcur[key] = dcur[key].to(device)
            
            # 3. 모델 추론
            # qDKT 등은 predict_one_step 사용
            if model_name in que_type_models and model_name not in ["lpkt", "promptkt"]:
                # predict_one_step은 보통 확률값을 반환함
                y = model.predict_one_step(dcur)
            else:
                # 일반 모델 폴백
                y = model(dcur)
                # 만약 Logit이라면 Sigmoid 필요할 수 있음 (모델 구현에 따라 다름)
                # 여기서는 predict_one_step 결과와 맞추기 위해 그대로 둠

            # 4. 데이터 추출 (마스킹 준비)
            qshft = dcur["shft_qseqs"]
            rshft = dcur["shft_rseqs"]
            sm = dcur["smasks"]

            # 5. 마스킹 및 리스트 저장
            # sm(Selection Mask)이 1인 유효 데이터만 추출
            mask = sm.bool()
            
            valid_qs = torch.masked_select(qshft, mask).cpu().numpy()
            valid_rs = torch.masked_select(rshft, mask).cpu().numpy()
            valid_ys = torch.masked_select(y, mask).cpu().numpy()
            
            for q, r, p in zip(valid_qs, valid_rs, valid_ys):
                results.append({
                    "question_id": int(q),
                    "true_label": int(r),
                    "pred_prob": float(p)
                })

    # CSV 저장
    if results:
        df_pred = pd.DataFrame(results)
        df_pred.to_csv(save_path, index=False)
        print(f"Extraction Done! Saved {len(df_pred)} rows.")
    else:
        print("Warning: No results extracted.")


def main(params):
    if params['use_wandb'] == 1:
        import wandb
        if "wandb_project_name" in params and params["wandb_project_name"] != "":
            wandb.init(project=params["wandb_project_name"])
        else:
            wandb.init()

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    # Config 로드
    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        
        # 불필요한 키 제거
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        
        # Emb_path 처리 (Data config 우선)
        if "emb_path" in model_config:
            del model_config["emb_path"]

        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name = trained_params["model_name"]
        dataset_name = trained_params["dataset_name"]
        emb_type = trained_params["emb_type"]
        
        if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt", "bakt_time", "sakt_que", "saint_que"]:
            train_config = config["train_config"]
            model_config["seq_len"] = train_config["seq_len"]

    # Data Config 로드
    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        
        # 모델별 특수 설정
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]

    # 사용자 입력 emb_path 적용
    if "emb_path" in trained_params:
        data_config["emb_path"] = trained_params["emb_path"]

    # 데이터셋 로더 초기화
    if model_name not in ["dimkt"]:
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    
    # 모델 로드
    # 주의: 만약 pykt 라이브러리를 수정하지 않았다면, NPY 로딩 시 에러가 날 수 있음.
    # 이 코드는 pykt/models/qdkt.py가 수정되었다고 가정하거나, 기본 로드 방식을 따름.
    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    # --- [핵심 기능 실행] 상세 예측 결과 저장 ---
    detailed_save_path = os.path.join(save_dir, "detailed_test_predictions.csv")
    save_detailed_predictions(model, test_loader, detailed_save_path, model_name)
    # ------------------------------------------

    # 기존 평가 로직 (선택적 실행)
    save_test_path = ""
    if params["save_all_preds"] == 1:
        save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")
        print(f"Predictions will be saved to {save_test_path}")

    if model.model_name == "rkt":
        testauc, testavg_prc, testacc = evaluate(model, test_loader, model_name, None, save_path=save_test_path)
    else:
        testauc, testavg_prc, testacc = evaluate(model, test_loader, model_name, save_path=save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    # Window 평가
    window_testauc, window_testacc = -1, -1
    save_test_window_path = ""
    if params["save_all_preds"] == 1:
        save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
    
    if model.model_name == "rkt":
        window_testauc, window_testavg_prc, window_testacc = evaluate(model, test_window_loader, model_name, None, save_path=save_test_window_path)
    else:
        window_testauc, window_testavg_prc, window_testacc = evaluate(model, test_window_loader, model_name, save_path=save_test_window_path)
    
    print(f"window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }

    # Question-level 평가 (있으면)
    q_testaucs, q_testaccs = -1, -1
    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_path=save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]
            
    print(dres)
    
    if params['use_wandb'] == 1:
        try:
            wandb.log(dres)
        except:
            pass

    # 결과 JSON 저장
    results_save_path = os.path.join(save_dir, "prediction_results.json")
    with open(results_save_path, 'w') as json_file:
        json.dump(dres, json_file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--save_all_preds", type=int, default=0)
    parser.add_argument("--wandb_project_name", type=str, default="")

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)