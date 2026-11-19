# Copyright 2024 Llamole Team
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

from typing import TYPE_CHECKING, List, Optional, Dict, Any

from ..data import get_dataset, DataCollatorForSeqGraph, get_template_and_fix_tokenizer
from ..extras.constants import IGNORE_INDEX, NO_LABEL_INDEX
from ..extras.misc import get_logits_processor
from ..extras.ploting import plot_loss
from ..model import load_tokenizer
from ..hparams import get_infer_args, get_train_args
from ..model import GraphLLMForCausalMLM
from .dataset import MolQADataset

import re
import os
import json
import math
import torch
import time
import signal
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )

def remove_extra_spaces(text):
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing spaces
    return cleaned_text.strip()

def run_eval(args: Optional[Dict[str, Any]] = None) -> None:
    print("Starting evaluation with args:", args)
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args)
    )

    if data_args.dataset in ["molqa", "molqa_drug", "molqa_material"]:
        run_molqa(
            model_args, data_args, training_args, finetuning_args, generating_args
        )
    else:
        raise ValueError("Unknown dataset: {}.".format(data_args.dataset))


def run_molqa(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
):
    tokenizer = load_tokenizer(model_args, generate_mode=True)["tokenizer"]

    data_info_path = os.path.join(data_args.dataset_dir, "dataset_info.json")
    with open(data_info_path, "r") as f:
        dataset_info = json.load(f)

    tokenizer.pad_token = tokenizer.eos_token
    dataset_name = data_args.dataset.strip()
    try:
        filename = dataset_info[dataset_name]["file_name"]
    except KeyError:
        raise ValueError(f"Dataset {dataset_name} not found in dataset_info.json")
    data_path = os.path.join(data_args.dataset_dir, f"{filename}")
    with open(data_path, "r") as f:
        original_data = json.load(f)

    # Create dataset and dataloader
    dataset = MolQADataset(original_data, tokenizer, data_args.cutoff_len)
    dataloader = DataLoader(
        dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False
    )

    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [
        tokenizer.eos_token_id
    ] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    model = GraphLLMForCausalMLM.from_pretrained(
        tokenizer, model_args, data_args, training_args, finetuning_args, load_adapter=True
    )

    all_results = []
    property_names = ["BBBP", "HIV", "BACE", "CO2", "N2", "O2", "FFV", "TC", "SC", "SA"]

    # Timeout and error handling settings
    max_samples = 50  # Limit to 50 samples
    timeout_per_sample = 120  # 2 minutes per sample
    max_retries = 3

    # Phase 1: Molecular Design
    global_idx = 0
    all_smiles = []
    samples_processed = 0

    def timeout_handler(signum, frame):
        raise TimeoutError("Sample evaluation timeout")

    for batch_idx, batch in enumerate(dataloader):
        if samples_processed >= max_samples:
            break
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        property_data = batch["property"].to(model.device)
        model.eval()
        with torch.no_grad():
            all_info_dict = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                molecule_properties=property_data,
                do_molecular_design=True,
                do_retrosynthesis=False,
                rollback=True,
                **gen_kwargs,
            )

            batch_results = []
            for i in range(len(all_info_dict["smiles_list"])):
                if samples_processed >= max_samples:
                    break

                original_data_idx = global_idx + i
                original_item = original_data[original_data_idx]

                # Timeout and error handling for each sample
                sample_start_time = time.time()
                sample_timeout = False

                try:
                    # Set up timeout handler
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_per_sample)

                    llm_response = "".join(item for item in all_info_dict["text_lists"][i])
                    result = {
                        "qa_idx": original_data_idx,
                        "instruction": original_item["instruction"],
                        "input": original_item["input"],
                        "llm_response": llm_response,
                        "response_design": remove_extra_spaces(llm_response),
                        "llm_smiles": all_info_dict["smiles_list"][i],
                        "property": {},
                        "processing_time": None,
                        "error": None,
                    }

                    # Add non-NaN property values
                    for j, prop_name in enumerate(property_names):
                        prop_value = property_data[i][j].item()
                        if not math.isnan(prop_value):
                            result["property"][prop_name] = prop_value

                    # Record processing time
                    result["processing_time"] = time.time() - sample_start_time
                    batch_results.append(result)
                    samples_processed += 1

                    # Cancel timeout
                    signal.alarm(0)

                except TimeoutError:
                    sample_timeout = True
                    print(f"Sample {original_data_idx} evaluation timeout after {timeout_per_sample} seconds")
                    result = {
                        "qa_idx": original_data_idx,
                        "instruction": original_item["instruction"],
                        "input": original_item["input"],
                        "llm_response": "TIMEOUT",
                        "response_design": "TIMEOUT",
                        "llm_smiles": None,
                        "property": {},
                        "processing_time": time.time() - sample_start_time,
                        "error": "TIMEOUT",
                    }
                    batch_results.append(result)
                    samples_processed += 1
                    signal.alarm(0)

                except Exception as e:
                    print(f"Error processing sample {original_data_idx}: {str(e)}")
                    result = {
                        "qa_idx": original_data_idx,
                        "instruction": original_item["instruction"],
                        "input": original_item["input"],
                        "llm_response": f"ERROR: {str(e)}",
                        "response_design": f"ERROR: {str(e)}",
                        "llm_smiles": None,
                        "property": {},
                        "processing_time": time.time() - sample_start_time,
                        "error": str(e),
                    }
                    batch_results.append(result)
                    samples_processed += 1
                    signal.alarm(0)

            all_results.extend(batch_results)
            all_smiles.extend([result['llm_smiles'] for result in batch_results])
            global_idx += len(batch_results)
    
    # Phase 2: Retrosynthesis
    retro_batch_start = 0
    for batch_idx, batch in enumerate(dataloader):
        if retro_batch_start >= len(all_results):
            break
        
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        batch_size = input_ids.shape[0]
        batch_smiles = all_smiles[retro_batch_start : retro_batch_start + batch_size]

        model.eval()
        with torch.no_grad():
            all_info_dict = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_molecular_design=False,
                do_retrosynthesis=True,
                input_smiles_list=batch_smiles,
                expansion_topk=50,
                iterations=100,
                max_planning_time=30,
                **gen_kwargs,
            )

            batch_results = []
            for i in range(batch_size):
                result = all_results[retro_batch_start + i]

                # Skip if already processed all required samples
                if retro_batch_start + i >= max_samples:
                    break

                # Timeout and error handling for retrosynthesis
                retro_start_time = time.time()
                retro_timeout = False

                try:
                    # Set up timeout handler for retrosynthesis
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_per_sample)

                    if result["llm_smiles"] is not None:
                        retro_plan = all_info_dict["retro_plan_dict"][result["llm_smiles"]]
                        result["llm_reactions"] = []
                        if retro_plan["success"]:
                            for reaction, template, cost in zip(
                                retro_plan["reaction_list"],
                                retro_plan["templates"],
                                retro_plan["cost"],
                            ):
                                result["llm_reactions"].append(
                                    {"reaction": reaction, "template": template, "cost": cost}
                                )

                    # new_text = "".join(item for item in all_info_dict["text_lists"][i])
                    if None in all_info_dict["text_lists"][i]:
                        print(f"List contains None: {all_info_dict['text_lists'][i]}")
                        new_text = "".join(item for item in all_info_dict["text_lists"][i] if item is not None)
                    else:
                        new_text = "".join(item for item in all_info_dict["text_lists"][i])

                    result["llm_response"] += new_text
                    result["llm_response"] = remove_extra_spaces(result["llm_response"])
                    result["response_retro"] = remove_extra_spaces(new_text)
                    result["retro_processing_time"] = time.time() - retro_start_time

                    # Cancel timeout
                    signal.alarm(0)

                except TimeoutError:
                    retro_timeout = True
                    print(f"Retrosynthesis for sample {retro_batch_start + i} timeout after {timeout_per_sample} seconds")
                    result["llm_reactions"] = []
                    result["response_retro"] = "RETRO_TIMEOUT"
                    result["retro_processing_time"] = time.time() - retro_start_time
                    result["retro_error"] = "TIMEOUT"
                    signal.alarm(0)

                except Exception as e:
                    print(f"Error in retrosynthesis for sample {retro_batch_start + i}: {str(e)}")
                    result["llm_reactions"] = []
                    result["response_retro"] = f"RETRO_ERROR: {str(e)}"
                    result["retro_processing_time"] = time.time() - retro_start_time
                    result["retro_error"] = str(e)
                    signal.alarm(0)

                batch_results.append(result)

            retro_batch_start += batch_size
    
    print(f'\nProcessed {len(all_results)} samples (limited to {max_samples})')
    print('all_results', all_results)

    print("\nSummary of results:")
    print_len = min(50, len(all_results))  # Show all results since max_samples=50

    # Count statistics
    timeout_count = sum(1 for r in all_results if r.get('error') == 'TIMEOUT')
    error_count = sum(1 for r in all_results if r.get('error') and r.get('error') != 'TIMEOUT')
    success_count = len(all_results) - timeout_count - error_count

    print(f"\nStatistics:")
    print(f"  Total samples processed: {len(all_results)}")
    print(f"  Successful: {success_count}")
    print(f"  Timeouts: {timeout_count}")
    print(f"  Errors: {error_count}")

    for result in all_results[:print_len]:
        print(f"\nData point {result['qa_idx']}:")
        print(f"  Instruction: {result['instruction']}")
        print(f"  Input: {result['input']}")
        print(f"  LLM Response: {result['llm_response']}")
        print(f"  LLM SMILES: {result['llm_smiles']}")
        print(f"  Number of reactions: {len(result.get('llm_reactions', []))}")
        print(f"  Processing time: {result.get('processing_time', 'N/A'):.2f}s")
        print(f"  Retro processing time: {result.get('retro_processing_time', 'N/A'):.2f}s")
        if result.get('error'):
            print(f"  Error: {result['error']}")
        if result.get('retro_error'):
            print(f"  Retro Error: {result['retro_error']}")
        for prop_name, prop_value in result["property"].items():
            print(f"  {prop_name}: {prop_value}")

    print(f"\nEvaluation completed. Processed {len(all_results)} samples with {timeout_per_sample}s timeout per sample.")