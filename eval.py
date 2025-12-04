import json
import pickle
from argparse import ArgumentParser
from typing import List, Dict, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from transformers import CLIPTextModelWithProjection
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from data_utils import collate_fn, PROJECT_ROOT, targetpad_transform
from loader import  MacirDataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF, encode_with_pseudo_tokens_HF_without
from models import build_text_encoder, Phi, PIC2WORD
from utils import extract_image_features, device, extract_pseudo_tokens_with_phi, extract_pseudo_tokens_without_phi
torch.multiprocessing.set_sharing_strategy('file_system')



@torch.no_grad()
def macir_generate_test_predictions(clip_model, query_test_dataset: Dataset, ref_names_list: List[str],
                                  pseudo_tokens: torch.Tensor, eval_type: str) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generates features predictions for the test set of MACIR
    """

    # Define the dataloader
    query_test_loader = DataLoader(dataset=query_test_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn)
    predicted_features_list = []
    target_names_list = []
    reference_names_list = []
    composition_type_list=[]
    condition_type_list=[]

    for batch in tqdm(query_test_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        condition_type=batch['condition_type']
        composition_type=batch['composition_type']
   

        batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        if eval_type=="text" :
            input_captions = [
                    f"{rel_caption}" for rel_caption in relative_captions]
            tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(device)
            text_features = encode_with_pseudo_tokens_HF(clip_model, tokenized_input_captions, batch_tokens)
            predicted_features = F.normalize(text_features)

        elif eval_type=="image":
            predicted_features = batch_tokens
        
        elif eval_type=="text_image":
            image_features=batch_tokens
            input_captions = [
                    f"{rel_caption}" for rel_caption in relative_captions]
            tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(device)
            text_features = encode_with_pseudo_tokens_HF(clip_model, tokenized_input_captions, batch_tokens)
            predicted_features = image_features + text_features
        
                
        else: #ZS-CIR (LinCIR, Pic2word, Searle, RTD)
            input_captions = [ f"a photo of $ that {rel_caption}" for rel_caption in relative_captions]
            tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(device)
            text_features = encode_with_pseudo_tokens_HF(clip_model, tokenized_input_captions, batch_tokens)
            predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)
        reference_names_list.extend(reference_names)
        composition_type_list.extend(composition_type)
        condition_type_list.extend(condition_type)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, reference_names_list, target_names_list, composition_type_list, condition_type_list



import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from torch.utils.data import Dataset


@torch.no_grad()
def macir_compute_test_metrics(
    query_test_dataset: Dataset,
    clip_model,
    index_features: torch.Tensor,
    index_names: List[str],
    ref_names_list: List[str],
    pseudo_tokens: torch.Tensor,
    eval_level: str,
    eval_type: str,
    split: str,  
) -> Dict[str, float]:
    """
    Compute retrieval metrics on the MACIR test set given:
      - the query dataset
      - index database features + names
      - pseudo tokens (when using projection-based ZS-CIR)
      - reference names / composition types / condition types
    Returns:
        metrics: dict mapping metric name -> value
    """
    
    # Generate predicted query features and metadata from the dataset
    predicted_features, reference_names, target_names, composition_types, condition_types = \
         macir_generate_test_predictions(clip_model, query_test_dataset, ref_names_list, pseudo_tokens, eval_type)

    # Determine device from index_features (or predicted_features)
    device = index_features.device

    # Move tensors to the target device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize features (index features; queries kept as they are)
    # index_features: (N, D)
    index_features = F.normalize(index_features, dim=-1).float()
    predicted_features = predicted_features.float()

    
    # Compute distances and sort
    # Let B = number of queries, N = size of the database
    # distances: (B, N)
    distances = 1 - predicted_features @ index_features.T
    # sorted_indices[b] = indices of index_features sorted by ascending distance
    sorted_indices = torch.argsort(distances, dim=-1).cpu()  # (B, N)
    # sorted_index_names_matrix[b, j] = index_names[ sorted_indices[b, j] ]
    sorted_index_names_matrix = np.array(index_names)[sorted_indices]  # (B, N)

    
    # Build reference prefixes for "restricted" evaluation cases
    reference_prefixes = [
        ref.split("_edit_")[0] if "_edit_" in ref else ref.split(".")[0]
        for ref in reference_names
    ]

    
    # Remove the reference (original) image itself from the ranking (we do not want the reference image to count as a retrieval candidate)
    # reference_names: (B,)
    # sorted_index_names_matrix: (B, N)
    # Build a mask that is False where the candidate equals the reference image,
    reference_mask = torch.tensor(
        sorted_index_names_matrix
        != np.repeat(np.array(reference_names), len(index_names)).reshape(
            len(target_names), -1
        )
    )
    # Apply the mask and reshape back to (B, N-1).
    # Now each row corresponds to candidates excluding the exact reference image.
    sorted_index_names_matrix = sorted_index_names_matrix[reference_mask].reshape(
        sorted_index_names_matrix.shape[0], sorted_index_names_matrix.shape[1] - 1
    )

    
    # Metric containers & type lists
    metrics = {}
    unique_composition_types = ['left_right', 'top_bottom', 'spatial_reasoning', 'size', 'action', 'color','object_reasoning','naive_object']
    unique_condition_types = ['remove', 'replace', 'add']
    # =========================================================================
    # CASE 1: eval_level == "full" (include all hard negatives)
    #   - Only group by composition_type
    # =========================================================================
    if eval_level == "full":
        for comp_type in unique_composition_types:
            # Indices of queries with the given composition_type
            type_indices = [
                i
                for i, ct in enumerate(composition_types)
                if ct == comp_type
            ]
            if not type_indices:
                # No queries with this composition_type
                continue

            # Select the rows corresponding to the chosen composition_type
            #   filtered_sorted_index_names_matrix: (B_c, N-1)
            filtered_sorted_index_names_matrix = sorted_index_names_matrix[type_indices]
            filtered_target_names = [target_names[i] for i in type_indices]

            # Build labels: labels[b, j] = True if candidate matches the target
            labels = torch.tensor(filtered_sorted_index_names_matrix== np.repeat(
                    np.array(filtered_target_names),
                    len(index_names) - 1,
                ).reshape(len(filtered_target_names), -1)
            )

            # Sanity check: For each query, exactly one candidate should match
            assert torch.equal(
                torch.sum(labels, dim=-1).int(),
                torch.ones(len(filtered_target_names)).int(),
            ), "Each query should have exactly one matching candidate."

            # Recall@1: fraction of queries where rank-1 candidate is correct
            recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100.0

            metrics[f"{comp_type}"] = {
                "recall_at1": recall_at1,
            }

        return metrics

    # =========================================================================
    # CASE 2: eval_level == "full_splits" (include all hard negatives, but split with condition_type)
    #   - Group by composition_type AND condition_type
    # =========================================================================
    elif eval_level == "full_splits":
        for comp_type in unique_composition_types:
            for cond_type in unique_condition_types:
                # Indices of queries matching both comp_type and cond_type
                type_indices = [
                    i
                    for i, (ct, cd) in enumerate(
                        zip(composition_types, condition_types)
                    )
                    if ct == comp_type and cd == cond_type
                ]

                if not type_indices:
                    # (e.g., spatial_reasoning+add, size+add)
                    metrics[f"{comp_type}_{cond_type}"] = {
                        "recall_at1": 0.0,
                    }
                    continue

                # Select the rows corresponding to this (comp_type, cond_type)
                filtered_sorted_index_names_matrix = sorted_index_names_matrix[
                    type_indices
                ]
                filtered_target_names = [target_names[i] for i in type_indices]

                # Build labels: labels[b, j] = True if candidate matches target
                labels = torch.tensor(
                    filtered_sorted_index_names_matrix
                    == np.repeat(
                        np.array(filtered_target_names),
                        len(index_names) - 1,
                    ).reshape(len(filtered_target_names), -1)
                )

                # Sanity check: one matching candidate per query
                assert torch.equal(
                    torch.sum(labels, dim=-1).int(),
                    torch.ones(len(filtered_target_names)).int(),
                ), "Each query should have exactly one matching candidate."

                # Recall@1 for this combination
                recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100.0

                metrics[f"{comp_type}_{cond_type}"] = {
                    "recall_at1": recall_at1,
                }

        return metrics

    # =========================================================================
    # CASE 3: eval_level is something else (e.g., "restricted"-style evaluation)
    #   - cond_type is given by `split` arg; we group by composition_type and
    #     restrict to candidates sharing the same "reference prefix"
    # =========================================================================
    elif eval_level == "restricted":
        cond_type = split 
        # -------------------------------------------------------------
        # For each query, keep only candidates that share the same
        #      prefix as the reference 
        # -------------------------------------------------------------
        reference_mask_split = torch.tensor(
            [
                [name.startswith(prefix) for name in sorted_index_names_row]
                for sorted_index_names_row, prefix in zip(
                    sorted_index_names_matrix, reference_prefixes
                )
            ],
            dtype=torch.bool,
        )

        # For each query, select only the candidates with the same prefix.
        # NOTE: This becomes a "ragged" list: each row can have different length. (some queries have multiple candidates (more than 2) )
        filtered_sorted_index_names_per_query = [
            np.array(sorted_index_names_row)[mask_row].tolist()
            for sorted_index_names_row, mask_row in zip(
                sorted_index_names_matrix, reference_mask_split
            )
        ]

        # -------------------------------------------------------------
        # Group by composition_type and compute recall@1
        #      using the filtered candidates
        # -------------------------------------------------------------
        for comp_type in unique_composition_types:
            # Indices of queries with the given composition_type
            type_indices = [
                i
                for i, ct in enumerate(composition_types)
                if ct == comp_type
            ]
            if not type_indices:
                # No queries for this composition type
                continue

            # Subset the filtered candidate lists and targets for this comp_type
            cur_rows = [filtered_sorted_index_names_per_query[i] for i in type_indices]
            cur_targets = [target_names[i] for i in type_indices]

            # Keep only queries that have at least 2 candidates
            # (we are interested in non-trivial cases)
            rows_targets = [
                (row, tgt)
                for row, tgt in zip(cur_rows, cur_targets)
                if len(row) > 1
            ]

            if not rows_targets:
                # No valid rows for this comp_type under the split setting
                continue

            cur_rows, cur_targets = zip(*rows_targets)  # unpack

            # Build labels: labels[q][j] = 1 if candidate j == target, else 0
            labels = [
                [int(name == target) for name in sorted_row]
                for sorted_row, target in zip(cur_rows, cur_targets)
            ]

            # Compute Recall@1: fraction of queries where the first candidate is correct
            num_queries = len(labels)
            num_correct_at1 = sum(1 for row in labels if row[0] == 1)
            recall_at1 = (num_correct_at1 / num_queries) * 100.0

            metrics[f"{comp_type}_{cond_type}"] = {
                "recall_at1": recall_at1,
            }

        return metrics


    
@torch.no_grad()
def macir_test_retrieval(dataset_path: str, image_encoder, text_encoder, split, ref_names_list: list, pseudo_tokens: torch.Tensor,
                       preprocess: callable,  eval_type: str, eval_level: str) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the MACIR test set given the pseudo tokens and the reference names
    """

    # Load the model
    # Extract the index features
    classic_test_dataset = MacirDataset(dataset_path, split, 'database', preprocess, eval_type,eval_level)
    index_features, index_names = extract_image_features(classic_test_dataset, image_encoder)
    # Define the query test dataset
    query_test_dataset = MacirDataset(dataset_path, split, 'query', preprocess, eval_type,eval_level)
       
    return macir_compute_test_metrics(query_test_dataset, text_encoder, index_features, index_names,
                                    ref_names_list, pseudo_tokens, eval_level, eval_type, split)



def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--eval-type", type=str,  required=True,
                        help=
                             "if 'phi' predicts the pseudo tokens using the phi network (LinCIR, SEARLE), "
                             "if 'pic2word' uses the pre-trained pic2word model to predict the pseudo tokens, "
                             "if 'image' or 'text' or 'image_text', is selected,  image/text embeddings or summation of them is used for retrieval"
                        )
    parser.add_argument("--split", default="remove",  choices=['add', 'remove', 'replace'], type=str, help="condition type split for restricted evaluation")
    parser.add_argument("--eval-level",  default="full", type=str, choices=['full', 'full_splits', 'restricted'], required=True, help="eval_level")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", required=True)

    parser.add_argument("--preprocess-type", default="clip", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument("--phi-checkpoint-name", type=str,
                        help="Phi checkpoint to use, needed when using phi, e.g. 'phi_20.pt'")
    parser.add_argument("--clip-model-name", default="large", type=str)
    parser.add_argument("--cache-dir", default="./hf_models", type=str)

    parser.add_argument("--l2-normalize", action="store_true", help="Whether or not to use l2 normalization")
    parser.add_argument("--clip-model-path", type=str, help="clip model path'")

    parser.add_argument("--text-encoder-checkpoint-name", type=str,
                        help="text checkpoint to use, needed when using RTD, e.g. 'text_encoder_best.pt'")
    parser.add_argument("--image-encoder-checkpoint-name", type=str,
                        help="image checkpoint to use, e.g. 'image_encoder_best.pt'")

    args = parser.parse_args()


    if args.eval_type == 'pic2word':
        args.mixed_precision = 'fp16'
        image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args)
        # need for RTD
        if args.text_encoder_checkpoint_name:
            text_encoder.load_state_dict(
                torch.load(args.text_encoder_checkpoint_name, map_location=device)[
                text_encoder.__class__.__name__])
            text_encoder=text_encoder.eval()
        
        phi = PIC2WORD(embed_dim=text_encoder.config.projection_dim,
                        output_dim=text_encoder.config.hidden_size,
                        ).to(device)
        sd = torch.load(args.phi_checkpoint_name, map_location=device)['state_dict_img2text']
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        phi.load_state_dict(sd)
        phi = phi.eval()
    else:
        args.mixed_precision = 'fp16'
        image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args)
        # need for RTD
        if args.text_encoder_checkpoint_name:
            text_encoder.load_state_dict(
                torch.load(args.text_encoder_checkpoint_name, map_location=device)[
                text_encoder.__class__.__name__])
            text_encoder=text_encoder.eval()
        if args.image_encoder_checkpoint_name:
            image_encoder.load_state_dict(
                torch.load(args.image_encoder_checkpoint_name, map_location=device)[
                image_encoder.__class__.__name__])
            image_encoder=image_encoder.eval()
        if args.eval_type == 'phi':
            phi = Phi(input_dim=text_encoder.config.projection_dim,
                        hidden_dim=text_encoder.config.projection_dim * 4,
                        output_dim=text_encoder.config.hidden_size, dropout=0.5).to(
                device)

            if args.phi_checkpoint_name:
                phi.load_state_dict(
                        torch.load(args.phi_checkpoint_name, map_location=device)[
                        phi.__class__.__name__])

            phi = phi.eval()
        else:
            phi=None
    
        
    if args.preprocess_type == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
    elif args.preprocess_type == 'clip':
        print('CLIP preprocess pipeline is used')
        preprocess = clip_preprocess
    else:
        raise ValueError("Preprocess type not supported")

        
    query_test_dataset = MacirDataset(args.dataset_path, args.split, 'query',  preprocess, args.eval_type, args.eval_level)
    
    image_encoder = image_encoder.float().to(device)
    text_encoder = text_encoder.float().to(device)
    if args.eval_type=="image" or args.eval_type=="text" or args.eval_type=="text_image":
        pseudo_tokens, ref_names_list = extract_pseudo_tokens_without_phi(image_encoder,  query_test_dataset, args)
    else:    
        pseudo_tokens, ref_names_list = extract_pseudo_tokens_with_phi(image_encoder, phi, query_test_dataset, args)
    pseudo_tokens = pseudo_tokens.to(device)
    
    
    print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")
    macir_metrics = macir_test_retrieval(args.dataset_path, image_encoder, text_encoder, args.split, ref_names_list, pseudo_tokens,
                                    preprocess,args.eval_type, args.eval_level)
        
   
    

    for k, v in macir_metrics.items():
        if isinstance(v, dict):  # Check if the value is a dictionary
            for _, sub_v in v.items():  # Iterate over the nested dictionary
                print(f"{k} = {sub_v:.1f}")
        else:  # If the value is not a dictionary
            print(f"{k} = {v:.1f}")


    # Specify the output file name    
    output_file = f"results_{args.eval_level}.txt"  
    with open(output_file, "w") as file:
        for k, v in macir_metrics.items():
            if isinstance(v, dict):  # Handle nested dictionaries for composition-condition metrics
                for sub_k, sub_v in v.items():
                    file.write(f"{sub_v:.2f},")
            else:  
                file.write(f"{k} = {v:.2f}\n")


if __name__ == '__main__':
    main()
