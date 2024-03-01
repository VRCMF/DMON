#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ast import literal_eval
import itertools as it
import time

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, check_min_version, get_full_repo_name, send_example_telemetry
from bert_graph_crop import BertForMultipleChoice, BertForMultipleChoice_full_map
from sklearn.metrics import f1_score, classification_report
from transformers.adapters import MAMConfig, LoRAConfig, UniPELTConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import copy
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def conflict_detect(predictions, args):
    # import numpy as np
    num_sentences = int(len(predictions)**0.5)
    # if args.dataset_domain == 'cdcp':
    #     logits_mtx = np.array(logits).reshape(num_sentences, num_sentences, 2)
    # else:
    #     logits_mtx = np.array(logits).reshape(num_sentences, num_sentences, 3)
    predictions_mtx = np.array(predictions).reshape(num_sentences, num_sentences)
    
    count = 0
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            # conflits detection
            if predictions_mtx[i, j] != 0 and predictions_mtx[j, i] != 0:
                count += 1
                # if logits_mtx[i, j]:
                #     pass

    return count

def post_process_diag(predictions):
    
    num_sentences = int(len(predictions)**0.5)
    predictions_mtx = predictions.reshape(num_sentences, num_sentences)

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i == j:
                predictions_mtx[i, j] = 0

    return predictions_mtx.view(-1)

def max_vote(logits1, logits2, pred1, pred2):

    pred1 = post_process_diag(pred1)
    pred2 = post_process_diag(pred2)
    pred_res = []
    confidence_res = []
    for i in range(len(logits1)):

        soft_logits1 = torch.nn.functional.softmax(logits1[i]) # [[j] for j in range(logits1.shape[1])]
        soft_logits2 = torch.nn.functional.softmax(logits2[i])

        # two class
        # torch.topk(soft_logits1, n=2)
        values_1, _ = soft_logits1.topk(k=2)
        values_2, _ = soft_logits2.topk(k=2)
        # import ipdb
        # ipdb.set_trace()
        # if (values_1[0] - values_2[0]) > (values_1[1] - values_2[1]):
        #     pred_res.append(int(pred1[i].detach().cpu().numpy()))
        # else:
        #     pred_res.append(int(pred2[i].detach().cpu().numpy()))
        if (values_1[0] - values_1[1]) >= (values_2[0] - values_2[1]):
            pred_res.append(int(pred1[i].detach().cpu().numpy()))
            confidence_res.append(float((values_1[0] - values_1[1]).detach().cpu().numpy()))
        else:
            pred_res.append(int(pred2[i].detach().cpu().numpy()))
            confidence_res.append(float((values_2[0] - values_2[1]).detach().cpu().numpy()))

    return pred_res, confidence_res

# def max_vote(logits1, logits2, pred1, pred2):

#     pred1 = post_process_diag(pred1)
#     pred2 = post_process_diag(pred2)
#     pred_res = []
#     for i in range(len(logits1)):

#         soft_logits1 = torch.nn.functional.softmax(logits1[i]) # [[j] for j in range(logits1.shape[1])]
#         soft_logits2 = torch.nn.functional.softmax(logits2[i])

#         # two class
#         # torch.topk(soft_logits1, n=2)
#         values_1, _ = soft_logits1.topk(k=2)
#         values_2, _ = soft_logits2.topk(k=2)
#         # import ipdb
#         # ipdb.set_trace()
#         # if (values_1[0] - values_2[0]) > (values_1[1] - values_2[1]):
#         #     pred_res.append(int(pred1[i].detach().cpu().numpy()))
#         # else:
#         #     pred_res.append(int(pred2[i].detach().cpu().numpy()))
#         if (values_1[0] - values_1[1]) >= (values_2[0] - values_2[1]):
#             pred_res.append(int(pred1[i].detach().cpu().numpy()))
#         else:
#             pred_res.append(int(pred2[i].detach().cpu().numpy()))

#     return pred_res

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    # MS-CNN
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--voter_branch",
        type=str,
        default="dual",
        help="The scheduler type to use.",
        choices=["head", "tail", "dual"],
    )
    parser.add_argument("--model_mode", type=str, default="bert_mtl_1d", choices=['bert_mtl_1d', 'bert_2d', 'bert_1d', 'bert', 'bert_self'])
    parser.add_argument("--dataset_domain", type=str, default="cdcp", choices=['cdcp', 'absRCT', 'ukp'])
    parser.add_argument("--win_size", type=int, default=12, help="symmetry cropping")
    parser.add_argument(
        "--full_map",
        action="store_true",
        help="input full map",
    )
    parser.add_argument(
        "--destroy",
        action="store_true",
        help="destroy symmetry matrix.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        argument_len = [feature.pop('pair_len') for feature in features]
        flattened_features = [
            [{k: v[i] for k, v in features[feature_i].items()} for i in range(argument_len[feature_i])] for feature_i in range(len(features))
        ]
        # {k: [list(it.islice(v, sl)) for sl in pair_len] for k, v in tokenized_examples.items()}
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        # batch = {k: v.view(batch_size, argument_len, -1) for k, v in batch.items()}
        # how to fix it ????
        batch = {k: v.view(batch_size, argument_len[0], -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    args = parse_args()
    args.cache_dir = ""
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_swag_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            if args.dataset_domain == 'cdcp':
                data_files["validation"] = args.validation_file
            elif args.dataset_domain == 'ukp':
                data_files["validation"] = args.validation_file
            else:
                data_files["validation_neo"] = args.validation_file
                data_files["validation_mix"] = args.validation_file.replace('neoplasm', 'mixed')
                data_files["validation_gla"] = args.validation_file.replace('neoplasm', 'glaucoma')
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    label_column_name = "label" if "label" in column_names else "labels"

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    config.win_size = args.win_size
    config.model_mode = args.model_mode
    config.dataset_domain = args.dataset_domain
    config.voter_branch = args.voter_branch
    config.destroy = args.destroy
    config.full_map = args.full_map
    # config['model_mode'] = config['win_size']
    # config['dataset_domain'] = config['win_size']
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if config.full_map:
            model = BertForMultipleChoice_full_map.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            )
        else:

            model = BertForMultipleChoice.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)


    # adapter_config = LoRAConfig(r=8, alpha=16)
    # # adapter_config = UniPELTConfig()
    # model.add_adapter("lora_adapter", config=adapter_config)
    # model.train_adapter("lora_adapter")
    # model.add_adapter("unipelt", config=adapter_config)
    # model.train_adapter("unipelt")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function_exp(examples):

        labels = [literal_eval(item) for item in examples['labels']]

        # Flatten out
        pair_list = [literal_eval(item) for item in examples['pairs']] 

        pair_len = [len(item) for item in pair_list]

        # flattened_features = list(chain(*flattened_features))


        first_sentences = []
        second_sentences = []
        for line_list in pair_list:
            for line in line_list:
                sent_item = line.strip().split('\t')
                first_sentences.append(sent_item[0].strip())
                second_sentences.append(sent_item[1].strip())
        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        # tokenized_inputs = {k: [v[i : i + pair_len[0]] for i in range(0, len(v), pair_len[0])] for k, v in tokenized_examples.items()}
        tokenized_inputs = {}
        for k, v in tokenized_examples.items():
            flatten_list = []
            head_idx = 0
            tail_idx = 0
            for pair_idx in pair_len:
                tail_idx = head_idx + pair_idx
                flatten_list.append(v[head_idx: tail_idx])
                head_idx = copy.copy(tail_idx)
            tokenized_inputs[k] = flatten_list

        tokenized_inputs["labels"] = labels
        tokenized_inputs["pair_len"] = pair_len
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function_exp, batched=True, remove_columns=raw_datasets["train"].column_names
        )

    train_dataset = processed_datasets["train"]
    if args.dataset_domain == 'cdcp':
        eval_dataset = processed_datasets["validation"]
    elif args.dataset_domain == 'ukp':
        eval_dataset = processed_datasets["validation"]
    else:
        eval_dataset_neo = processed_datasets["validation_neo"]
        eval_dataset_mix = processed_datasets["validation_mix"]
        eval_dataset_gla = processed_datasets["validation_gla"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    if args.dataset_domain == 'cdcp':
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    elif args.dataset_domain == 'ukp':
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    else:
        eval_dataloader_neo = DataLoader(eval_dataset_neo, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        eval_dataloader_mix = DataLoader(eval_dataset_mix, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        eval_dataloader_gla = DataLoader(eval_dataset_gla, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # for n, p in model.named_parameters():
    #     if 'bert' in n:
    #         p.requires_grad = False
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.dataset_domain == 'cdcp':
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    elif args.dataset_domain == 'ukp':
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    else:
        model, optimizer, train_dataloader, eval_dataloader_neo, eval_dataloader_mix, eval_dataloader_gla, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader_neo, eval_dataloader_mix, eval_dataloader_gla, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("swag_no_trainer", experiment_config)

    # Metrics
    # metric = evaluate.load("f1")
    # metric1 = evaluate.load("f1")
    # metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    if args.dataset_domain == 'cdcp':
        fscore = 0
    elif args.dataset_domain == 'ukp':
        fscore = 0
    else:
        fscore_neo = 0
        fscore_mix = 0
        fscore_gla = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        if epoch == 0:
            model_dir = os.path.join('', '_'.join(['{}_{}'.format(args.model_mode, args.dataset_domain), time.strftime('%b_%d_%H_%M_%S', time.localtime())]))
            os.makedirs(model_dir)
        with open(model_dir + '/desc.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join([str(args.model_name_or_path), str(args.description), str(args.dataset_domain), str(args.model_mode)]))
            f.close()
        model.train()
        if args.with_tracking:
            total_loss = 0
        train_start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                batch['mode'] = True
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        train_end_time = time.time()
        print('train finish in {}'.format(train_end_time - train_start_time))
        model.eval()
        
        # eval_dataloader_neo, eval_dataloader_mix, eval_dataloader_gla
        # --------------------------------------------------------
        if args.dataset_domain == 'cdcp':
            p_list = []
            r_list = []
            conflicts_num = 0
            eval_start_time = time.time()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    batch['mode'] = False
                    outputs = model(**batch)
            #     predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
            #     p_list += predictions
            #     r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
            
            # print('#'*10)
            # print(classification_report(r_list, p_list, digits=4))
            # print('#'*10)
                if args.model_mode == 'bert_mtl_1d':
                    # predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                    if args.voter_branch == "dual":
                        predictions, scores = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                        mtx_len = int(len(predictions)**0.5)
                        # import ipdb
                        # ipdb.set_trace()
                        pred_mtx = np.array(predictions).reshape(mtx_len, mtx_len)
                        score_mtx = np.array(scores).reshape(mtx_len, mtx_len)
                        for i in range(mtx_len):
                            for j in range(mtx_len):
                                # if pred_mtx[i,j] == 1 and pred_mtx[j,i] == 2:
                                #     pass
                                # elif pred_mtx[i,j] == 1 and pred_mtx[j,i] == 1:
                                if pred_mtx[i, j] == 2:
                                    pred_mtx[j, i] = 1

                        # import ipdb
                        # ipdb.set_trace()
                    elif args.voter_branch == "head":
                        predictions = list(outputs.logits[0].argmax(dim=-1).detach().cpu().numpy())
                    else:
                        predictions = list(outputs.logits[1].argmax(dim=-1).detach().cpu().numpy())
                    # import ipdb
                    # ipdb.set_trace()
                    p_list += predictions
                else:
                    predictions = outputs.logits.argmax(dim=-1)
                    p_list += list(predictions.detach().cpu().numpy())
                # import ipdb
                # ipdb.set_trace()
                r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
                # conflicts_num += conflict_detect(predictions, args)
            
            eval_end_time = time.time()
            print('eval finish in {}'.format(eval_end_time - eval_start_time))

            # print(classification_report(r_list, p_list, digits=4))
            # print('#'*10)
            print('Detect {} conflicts'.format(str(conflicts_num)))


            # with open(model_dir + '/conflits.txt', 'w+', encoding='utf-8') as f:
            #     out_str = ''.join([str(conflicts_num)])
            #     out_str += '\n'
            #     f.write(out_str)
            #     f.close()

            # sep_str = ''.join(classification_report(r_list, p_list, digits=4))
            r_list = [1 if ele != 0 else 0 for ele in r_list]
            p_list = [1 if ele != 0 else 0 for ele in p_list]
            print(classification_report(r_list, p_list, digits=4))
            print('#'*10)
            if f1_score(r_list, p_list, average='macro') > fscore:
                fscore = f1_score(r_list, p_list, average='macro')
                torch.save(model.state_dict(), model_dir + '/best.pth')
                with open(model_dir + '/results.txt', 'w', encoding='utf-8') as f:
                    # out_str = sep_str + '\n' + ''.join(classification_report(r_list, p_list, digits=4))
                    out_str = ''.join(classification_report(r_list, p_list, digits=4))
                    out_str += '\n'
                    # out_str += '{}'.format(str(conflicts_num))
                    f.write(out_str)
                    f.close()
        elif args.dataset_domain == 'ukp':
            p_list = []
            r_list = []
            conflicts_num = 0
            eval_start_time = time.time()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    batch['mode'] = False
                    outputs = model(**batch)
            #     predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
            #     p_list += predictions
            #     r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
            
            # print('#'*10)
            # print(classification_report(r_list, p_list, digits=4))
            # print('#'*10)
                if args.model_mode == 'bert_mtl_1d':
                    # predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                    if args.voter_branch == "dual":
                        predictions, scores = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                        # mtx_len = int(len(predictions)**0.5)
                        # # import ipdb
                        # # ipdb.set_trace()
                        # pred_mtx = np.array(predictions).reshape(mtx_len, mtx_len)
                        # score_mtx = np.array(scores).reshape(mtx_len, mtx_len)
                        # for i in range(mtx_len):
                        #     for j in range(mtx_len):
                        #         # if pred_mtx[i,j] == 1 and pred_mtx[j,i] == 2:
                        #         #     pass
                        #         # elif pred_mtx[i,j] == 1 and pred_mtx[j,i] == 1:
                        #         if pred_mtx[i, j] == 2:
                        #             pred_mtx[j, i] = 1

                        # import ipdb
                        # ipdb.set_trace()
                    elif args.voter_branch == "head":
                        predictions = list(outputs.logits[0].argmax(dim=-1).detach().cpu().numpy())
                    else:
                        predictions = list(outputs.logits[1].argmax(dim=-1).detach().cpu().numpy())
                    # import ipdb
                    # ipdb.set_trace()
                    p_list += predictions
                else:
                    predictions = outputs.logits.argmax(dim=-1)
                    p_list += list(predictions.detach().cpu().numpy())
                # import ipdb
                # ipdb.set_trace()
                r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
                # conflicts_num += conflict_detect(predictions, args)
            
            eval_end_time = time.time()
            print('eval finish in {}'.format(eval_end_time - eval_start_time))

            # print(classification_report(r_list, p_list, digits=4))
            # print('#'*10)
            print('Detect {} conflicts'.format(str(conflicts_num)))


            # with open(model_dir + '/conflits.txt', 'w+', encoding='utf-8') as f:
            #     out_str = ''.join([str(conflicts_num)])
            #     out_str += '\n'
            #     f.write(out_str)
            #     f.close()

            # sep_str = ''.join(classification_report(r_list, p_list, digits=4))
            # r_list = [1 if ele != 0 else 0 for ele in r_list]
            # p_list = [1 if ele != 0 else 0 for ele in p_list]
            print(classification_report(r_list, p_list, digits=4))
            print('#'*10)
            if f1_score(r_list, p_list, average='macro') > fscore:
                fscore = f1_score(r_list, p_list, average='macro')
                torch.save(model.state_dict(), model_dir + '/best.pth')
                with open(model_dir + '/results.txt', 'w', encoding='utf-8') as f:
                    # out_str = sep_str + '\n' + ''.join(classification_report(r_list, p_list, digits=4))
                    out_str = ''.join(classification_report(r_list, p_list, digits=4))
                    out_str += '\n'
                    # out_str += '{}'.format(str(conflicts_num))
                    f.write(out_str)
                    f.close()
        else:
            p_list = []
            r_list = []
            conflicts_num_neo = 0
            eval_start_time = time.time()
            for step, batch in enumerate(eval_dataloader_neo):
                with torch.no_grad():
                    batch['mode'] = False
                    outputs = model(**batch)
                if args.model_mode == 'bert_mtl_1d':
                    # predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                    if args.voter_branch == "dual":
                        predictions, confidence_res = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                    elif args.voter_branch == "head":
                        predictions = list(outputs.logits[0].argmax(dim=-1).detach().cpu().numpy())
                    else:
                        predictions = list(outputs.logits[1].argmax(dim=-1).detach().cpu().numpy())
                    # import ipdb
                    # ipdb.set_trace()
                    p_list += predictions
                else:
                    predictions = outputs.logits.argmax(dim=-1)
                    p_list += list(predictions.detach().cpu().numpy())
                # import ipdb
                # ipdb.set_trace()
                r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
                # conflicts_num_neo += conflict_detect(predictions, args)
            
            eval_end_time = time.time()
            print('eval finish in {}'.format(eval_end_time - eval_start_time))
            # import ipdb
            # ipdb.set_trace()
            print(classification_report(r_list, p_list, digits=4))
            print('#'*10)
            print('Detect {} conflicts'.format(str(conflicts_num_neo)))
            #     if args.model_mode == 'bert_mtl_1d':
            #         predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
            #         p_list += predictions
            #     else:
            #         predictions = outputs.logits.argmax(dim=-1)
            #         p_list += list(predictions.detach().cpu().numpy())
            #     r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
            
            # print('#'*10)
            # print(classification_report(r_list, p_list, digits=4))
            # print('#'*10)
            fscore_neo_cur = f1_score(r_list, p_list, average='macro')
            
            # --------------------------------------------------------
            # --------------------------------------------------------
            p_list = []
            r_list = []
            conflicts_num_gla = 0
            for step, batch in enumerate(eval_dataloader_gla):
                with torch.no_grad():
                    batch['mode'] = False
                    outputs = model(**batch)
                if args.model_mode == 'bert_mtl_1d':
                    # predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                    if args.voter_branch == "dual":
                        predictions, confidence_res = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                    elif args.voter_branch == "head":
                        predictions = list(outputs.logits[0].argmax(dim=-1).detach().cpu().numpy())
                    else:
                        predictions = list(outputs.logits[1].argmax(dim=-1).detach().cpu().numpy())
                    # import ipdb
                    # ipdb.set_trace()
                    p_list += predictions
                else:
                    predictions = outputs.logits.argmax(dim=-1)
                    p_list += list(predictions.detach().cpu().numpy())
                r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
                # conflicts_num_gla += conflict_detect(predictions, args)
            
            print(classification_report(r_list, p_list, digits=4))
            print('#'*10)
            print('Detect {} conflicts'.format(str(conflicts_num_gla)))
            #     if args.model_mode == 'bert_mtl_1d':
            #         predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
            #         p_list += predictions
            #     else:
            #         predictions = outputs.logits.argmax(dim=-1)
            #         p_list += list(predictions.detach().cpu().numpy())
            #     r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
            
            # print('#'*10)
            # print(classification_report(r_list, p_list, digits=4))
            # print('#'*10)

            fscore_gla_cur = f1_score(r_list, p_list, average='macro')

            # --------------------------------------------------------
            # --------------------------------------------------------
            p_list = []
            r_list = []
            conflicts_num_mix = 0
            for step, batch in enumerate(eval_dataloader_mix):
                with torch.no_grad():
                    batch['mode'] = False
                    outputs = model(**batch)
                if args.model_mode == 'bert_mtl_1d':
                    # predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                    if args.voter_branch == "dual":
                        predictions, confidence_res = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
                    elif args.voter_branch == "head":
                        predictions = list(outputs.logits[0].argmax(dim=-1).detach().cpu().numpy())
                    else:
                        predictions = list(outputs.logits[1].argmax(dim=-1).detach().cpu().numpy())
                    # import ipdb
                    # ipdb.set_trace()
                    p_list += predictions
                else:
                    predictions = outputs.logits.argmax(dim=-1)
                    p_list += list(predictions.detach().cpu().numpy())
                r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
                # conflicts_num_mix += conflict_detect(predictions, args)
            
            print(classification_report(r_list, p_list, digits=4))
            print('#'*10)
            print('Detect {} conflicts'.format(str(conflicts_num_mix)))
            #     if args.model_mode == 'bert_mtl_1d':
            #         predictions = max_vote(outputs.logits[0], outputs.logits[1], outputs.logits[0].argmax(dim=-1), outputs.logits[1].argmax(dim=-1))
            #         p_list += predictions
            #     else:
            #         predictions = outputs.logits.argmax(dim=-1)
            #         p_list += list(predictions.detach().cpu().numpy())
            #     r_list += list(batch["labels"].squeeze(0).detach().cpu().numpy())
            
            # print('#'*10)
            # print(classification_report(r_list, p_list, digits=4))
            # print('#'*10)
            fscore_mix_cur = f1_score(r_list, p_list, average='macro')
            # --------------------------------------------------------
            # --------------------------------------------------------

            with open(model_dir + '/conflits.txt', 'w+', encoding='utf-8') as f:
                out_str = ''.join([str(conflicts_num_neo), str(conflicts_num_mix), str(conflicts_num_gla)])
                out_str += '\n'
                f.write(out_str)
                f.close()

            if (fscore_gla_cur + fscore_mix_cur + fscore_neo_cur) > (fscore_gla + fscore_mix + fscore_neo):
                fscore_gla = fscore_gla_cur
                fscore_mix = fscore_mix_cur
                fscore_neo = fscore_neo_cur
                torch.save(model.state_dict(), model_dir + '/best.pth')
                with open(model_dir + '/results.txt', 'w', encoding='utf-8') as f:
                    out_str = ''.join([str(fscore_neo_cur), str(fscore_gla_cur), str(fscore_mix_cur)])
                    out_str += '\n'
                    # out_str += ''.join([str(fscore_neo_cur), str(fscore_mix_cur), str(fscore_gla_cur)])
                    f.write(out_str)
                    f.close()
        

if __name__ == "__main__":
    main()
