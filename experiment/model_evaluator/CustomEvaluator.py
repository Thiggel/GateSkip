import re
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
from pathlib import Path


class CustomEvaluator:
    """
    Evaluator for pretrained models on GSM8K or CSQA using template-based prompting
    and answer extraction from tags.
    """

    def __init__(
        self,
        model,
        tokenizer,
        batch_size: int = 8,
        save_results: bool = False,
        results_dir: str = "./results",
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: The pretrained model to evaluate
            tokenizer: The tokenizer for the model
            batch_size: Batch size for evaluation
            save_results: Whether to save detailed results to disk
            results_dir: Directory to save results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        if self.save_results:
            self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Configure logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def format_prompt(self, question: str) -> str:
        """Format a question with the template"""
        return f"Question: {question}\nAnswer:"

    def extract_answer(self, generated_text: str) -> Optional[str]:
        """Extract the answer from the generated text using <answer></answer> tags"""
        print(generated_text)
        match = re.search(r"####\s*(.*?)(?:\n|$)", generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def normalize_answer(self, answer: str) -> str:
        """Normalize the answer for comparison"""
        # Remove whitespace, commas in numbers, and convert to lowercase
        normalized = re.sub(r"\s+", "", answer)
        normalized = re.sub(r"(\d),(\d)", r"\1\2", normalized)
        normalized = normalized.lower()
        return normalized

    def extract_numerical_answer(self, answer: str) -> Optional[float]:
        """Extract numerical value from an answer string"""
        # First try to find a number at the end of the string
        match = re.search(r"(\-?\d+\.?\d*)%?$", answer.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # Try to find any number in the string
        match = re.search(r"(\-?\d+\.?\d*)%?", answer.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None

    def compare_answers(self, predicted: str, reference: str, dataset_name: str) -> bool:
        """Compare predicted answer to reference answer"""
        if predicted is None:
            return False
            
        if dataset_name.lower() == "gsm8k":
            # For GSM8K, extract numerical answers and compare
            pred_num = self.extract_numerical_answer(predicted)
            ref_num = self.extract_numerical_answer(reference)
            
            if pred_num is not None and ref_num is not None:
                # Allow small floating point differences
                return abs(pred_num - ref_num) < 1e-6
            
            # Fall back to normalized string comparison
            return self.normalize_answer(predicted) == self.normalize_answer(reference)
        else:
            # For CSQA or other datasets, use normalized string comparison
            return self.normalize_answer(predicted) == self.normalize_answer(reference)

    def evaluate(
        self, 
        dataset_name: str, 
        split: str = "test",
        num_samples: Optional[int] = None,
        seed: int = 42,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the specified dataset.
        
        Args:
            dataset_name: Name of the dataset ('gsm8k' or 'csqa')
            split: Dataset split to evaluate on
            num_samples: Number of samples to evaluate (None for all)
            seed: Random seed for sampling
            generation_kwargs: Keyword arguments for model.generate()
            
        Returns:
            Dictionary containing evaluation results
        """
        # Set default generation kwargs if none provided
        if generation_kwargs is None:
            generation_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
            }
            
        # Load the dataset
        self.logger.info(f"Loading {dataset_name} dataset, {split} split")
        if dataset_name.lower() == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=split)
            question_key = "question"
            answer_key = "answer"
        elif dataset_name.lower() == "commonsense_qa":
            dataset = load_dataset("commonsense_qa", split=split)
            question_key = "question"
            answer_key = "answerKey"  # This is just the label, we'll handle it specially
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        # Sample if num_samples is specified
        if num_samples is not None and num_samples < len(dataset):
            np.random.seed(seed)
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = dataset.select(indices)
            
        self.logger.info(f"Evaluating on {len(dataset)} examples")
        
        # Process dataset and evaluate
        correct = 0
        total = 0
        results = []

        # Process in batches
        for i in tqdm(range(0, len(dataset), self.batch_size)):
            batch = dataset[i:i+self.batch_size]
            
            # Format prompts
            questions = batch[question_key]
            
            if dataset_name.lower() == "gsm8k":
                reference_answers = [item.split('#### ')[-1] for item in batch[answer_key]]
            else:  # CSQA
                # For CSQA, we need to handle the choices and labels
                reference_answers = []
                for item in batch:
                    choice_label = item[answer_key]  # A, B, C, D, or E
                    choices = item["choices"]
                    correct_idx = ord(choice_label) - ord('A')
                    if 0 <= correct_idx < len(choices):
                        correct_choice = choices[correct_idx]["text"]
                        reference_answers.append(correct_choice)
                    else:
                        # Fallback if something's wrong with the format
                        reference_answers.append(choice_label)
            
            prompts = [self.format_prompt(q) for q in questions]
            
            # Tokenize inputs
            self.tokenizer.padding_side = 'left'
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True).to(self.device)
            
            # Generate outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode outputs and extract answers
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predicted_answers = [self.extract_answer(text) for text in generated_texts]
            
            # Calculate accuracy for this batch
            for q, ref, pred, full_text in zip(questions, reference_answers, predicted_answers, generated_texts):
                print(full_text, "Prediction: ", pred, "Reference: ",ref, '\n\n\n')
                is_correct = self.compare_answers(pred, ref, dataset_name)
                
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    "question": q,
                    "reference_answer": ref,
                    "predicted_answer": pred,
                    "full_generation": full_text,
                    "is_correct": is_correct
                })
                
                # Log progress occasionally
                if total % 50 == 0:
                    self.logger.info(f"Current accuracy: {correct}/{total} = {correct/total:.4f}")
        
        # Calculate final metrics
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        self.logger.info(f"Final accuracy: {correct}/{total} = {accuracy:.4f}")

        print(metrics)
        
        # Save detailed results if requested
        if self.save_results:
            results_file = self.results_dir / f"{dataset_name}_{split}_{len(dataset)}_results.json"
            with open(results_file, "w") as f:
                json.dump({
                    "dataset": dataset_name,
                    "split": split,
                    "num_samples": len(dataset),
                    "metrics": metrics,
                    "generation_kwargs": generation_kwargs,
                    "results": results
                }, f, indent=2)
            self.logger.info(f"Saved detailed results to {results_file}")
        
        return {
            dataset_name: accuracy
        }
