import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np
from .data_loader import KTDataset
from .que_data_loader import KTQueDataset
from pykt.config import que_type_models

# Only KTDataset and KTQueDataset are used by the 5 Polya models
# Other dataloaders (dkt_forget, atdkt, lpkt, dimkt, cskt) removed


def init_test_datasets(data_config, model_name, batch_size, diff_level=None):
    dataset_name = data_config["dataset_name"]
    print(f"model_name is {model_name}, dataset_name is {dataset_name}")
    test_question_loader, test_question_window_loader = None, None

    # All 5 Polya models are in que_type_models
    if model_name in que_type_models:
        test_dataset = KTQueDataset(
            os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
            input_type=data_config["input_type"],
            folds=[-1],
            concept_num=data_config["num_c"],
            max_concepts=data_config["max_concepts"],
        )
        test_window_dataset = KTQueDataset(
            os.path.join(
                data_config["dpath"], data_config["test_window_file_quelevel"]
            ),
            input_type=data_config["input_type"],
            folds=[-1],
            concept_num=data_config["num_c"],
            max_concepts=data_config["max_concepts"],
        )
        test_question_dataset = None
        test_question_window_dataset = None
    else:
        # Fallback to KTDataset for any other models (shouldn't happen with 5 Polya models)
        test_dataset = KTDataset(
            os.path.join(data_config["dpath"], data_config["test_file"]),
            data_config["input_type"],
            {-1},
        )
        test_window_dataset = KTDataset(
            os.path.join(data_config["dpath"], data_config["test_window_file"]),
            data_config["input_type"],
            {-1},
        )
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(
                os.path.join(data_config["dpath"], data_config["test_question_file"]),
                data_config["input_type"],
                {-1},
                True,
            )
            test_question_window_dataset = KTDataset(
                os.path.join(
                    data_config["dpath"], data_config["test_question_window_file"]
                ),
                data_config["input_type"],
                {-1},
                True,
            )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(
        test_window_dataset, batch_size=batch_size, shuffle=False
    )
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader, test_question_window_loader = None, None
        if not test_question_dataset is None:
            test_question_loader = DataLoader(
                test_question_dataset, batch_size=batch_size, shuffle=False
            )
        if not test_question_window_dataset is None:
            test_question_window_loader = DataLoader(
                test_question_window_dataset, batch_size=batch_size, shuffle=False
            )

    return (
        test_loader,
        test_window_loader,
        test_question_loader,
        test_question_window_loader,
    )


def init_dataset4train(
    dataset_name,
    model_name,
    data_config,
    i,
    batch_size,
    diff_level=None,
    train_subset_rate=1.0,
):
    """Initialize training and validation datasets.

    Simplified for 5 BAIM models (akt_baim, qdkt_baim, simplekt_baim,
    qikt_baim, sparsekt_baim) which all use KTQueDataset.
    """
    print(f"dataset_name: {dataset_name}, model_name: {model_name}, fold: {i}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])

    # All 5 BAIM models use KTQueDataset (question-level datasets)
    if model_name in que_type_models:
        curvalid = KTQueDataset(
            os.path.join(
                data_config["dpath"], data_config["train_valid_file_quelevel"]
            ),
            input_type=data_config["input_type"],
            folds={i},
            concept_num=data_config["num_c"],
            max_concepts=data_config["max_concepts"],
        )
        curtrain = KTQueDataset(
            os.path.join(
                data_config["dpath"], data_config["train_valid_file_quelevel"]
            ),
            input_type=data_config["input_type"],
            folds=all_folds - {i},
            concept_num=data_config["num_c"],
            max_concepts=data_config["max_concepts"],
            subset_rate=train_subset_rate,
        )
    else:
        # Fallback to basic KTDataset (shouldn't be used with 5 Polya models)
        print(f"Warning: {model_name} not in que_type_models, using fallback KTDataset")
        curvalid = KTDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            {i},
        )
        curtrain = KTDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            all_folds - {i},
            subset_rate=train_subset_rate,
        )

    train_loader = DataLoader(curtrain, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(curvalid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
