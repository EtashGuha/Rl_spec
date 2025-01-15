import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import wandb
import argparse
from torch.utils.data import DataLoader
import json
import datetime
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
from tqdm import tqdm
from torch import optim

def sample_tokens(proposed_output, verified_output) -> torch.Tensor:
        # Accept-reject token loop
        accept_ids = []
        all_accepted = True
        sample_steps = 0
        alpha = 0
        num_tokens_accepted = 1
        for t in range(proposed_output.generated_len):
            sampled_ratios = (
                verified_output.output_distribution[t,
                                                    proposed_output.output_ids[0, t]]
                / proposed_output.output_distribution[t, proposed_output.output_ids[0, t]]
            )
            sampled_ratios = torch.min(sampled_ratios,
                                       torch.ones_like(sampled_ratios))
            rs = torch.rand_like(sampled_ratios)
            # logger.log("sample ratio", (rs, sampled_ratios))
            cur_alpha = min(verified_output.output_distribution[t, proposed_output.output_ids[0, t]],
                            proposed_output.output_distribution[t, proposed_output.output_ids[0, t]])

            assert cur_alpha >= 0 and cur_alpha <= 1
            alpha += cur_alpha
            sample_steps += 1

            if rs < sampled_ratios:
                accept_ids.append(proposed_output.output_ids[:, t])
                num_tokens_accepted += 1
            else:
                all_accepted = False
                if verified_output.output_distribution[t, :].shape != proposed_output.output_distribution[t, :]:
                    next_token_id = target_sample_from_distribution(
                        verified_output.output_distribution[t, :],
                        torch.zeros_like(verified_output.output_distribution[t, :]))
                else:
                    next_token_id = target_sample_from_distribution(
                        verified_output.output_distribution[t, :],
                        proposed_output.output_distribution[t, :])
                accept_ids.append(next_token_id.unsqueeze(0))
                break

        # if all tokens were accepted, sample a last one
        if all_accepted:
            next_token_id = torch.multinomial(
                verified_output.output_distribution[-1, :], num_samples=1)

            assert next_token_id.dim() == 1
            accept_ids.append(next_token_id)
        accept_ids = torch.cat(accept_ids, dim=0)
        return accept_ids.unsqueeze(0), alpha, sample_steps, num_tokens_accepted

@dataclass
class OutputAndCache:
    generated_len: int
    output_ids: torch.Tensor
    output_logits: torch.Tensor
    output_distribution: torch.Tensor
    past_key_values: Optional[torch.Tensor] = None

def target_sample_from_distribution(target_distribution, draft_distribution):
    distribution = (target_distribution - draft_distribution)
    distribution = torch.max(distribution,
                             torch.zeros_like(distribution))
    if (distribution.sum(dim=-1, keepdim=True) == 0).any():
        distribution = torch.where(
            distribution == 0, distribution + 1e-10, distribution)
        print("[Warning] Distribution contains zero values")
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    return torch.multinomial(distribution, num_samples=1).squeeze(-1)


class SpeculativeDecoder:
    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        tokenizer,
        max_new_tokens: int = 32,
        max_proposals_per_step: int = 4,
        device: str = "cuda",
        max_length: int = 512,
        num_rollouts: int = 5
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.max_proposals_per_step = max_proposals_per_step
        self.device = device
        self.num_rollouts = num_rollouts

    def generate(self, input_ids: torch.Tensor):
        """Generate text using speculative decoding with PPO-specific outputs"""
        if input_ids.size(0) != 1:
            raise ValueError("This implementation only supports batch size 1")

        current_input = input_ids.clone()
        draft_input = input_ids.clone()
        total_probs = []
        rewards = []
        with torch.no_grad():
            for _ in range(self.num_rollouts):
                total_generated_tokens = 0
                propose_steps = 0
                # Generate proposals from draft model

                scores = torch.tensor([]).cuda()
                for i in range(self.max_proposals_per_step):
                    output = self.draft_model(draft_input, output_hidden_states=True, return_dict=True)
                    if len(scores) == 0:
                        scores = output.logits[0][-1].unsqueeze(0)
                        draft_input = torch.cat([draft_input, torch.multinomial(F.softmax(output.logits[0][-1]), num_samples=1).unsqueeze(dim=0)], dim=-1)
                    else:
                        scores = torch.cat([scores, output.logits[0][-1].unsqueeze(0)])
                        try:
                            draft_input = torch.cat([draft_input, torch.multinomial(F.softmax(output.logits[0][-1]), num_samples=1).unsqueeze(dim=0)], dim=-1)
                        except:
                            breakpoint()
                draft_logits = scores
                draft_probs = F.softmax(draft_logits, dim=-1)       
                
                # Sample proposals
                proposed_tokens = torch.multinomial(draft_probs, num_samples=1)
                # Prepare inputs for target model
                target_input = torch.cat([current_input, proposed_tokens.T], dim=-1)
                
                # Run target model
                target_outputs = self.target_model(
                    target_input,
                    output_hidden_states=True,
                    return_dict=True
                )
                target_logits = target_outputs.logits[:, current_input.size(1)-1:-1, :]
                target_probs = F.softmax(target_logits, dim=-1)
                target_tokens = torch.multinomial(target_probs.squeeze(0), num_samples=1)
                draft_output_and_cache = OutputAndCache(output_ids=proposed_tokens.T, generated_len=len(proposed_tokens), output_logits=draft_logits, output_distribution=draft_probs)
                target_output_and_cache = OutputAndCache(output_ids=target_tokens.T, generated_len=len(target_tokens), output_logits=target_logits, output_distribution=target_probs.squeeze(0))

                accepted, alpha, sample_steps, num_tokens_accepted = sample_tokens(draft_output_and_cache, target_output_and_cache)
                probs = []
                for i in range(len(proposed_tokens)):
                    probs.append(draft_probs[i, proposed_tokens[i]])
                total_probs.append(torch.prod(torch.tensor(probs)))
                current_input = torch.cat([current_input, accepted], dim=-1)
                propose_steps += 1

                total_generated_tokens += num_tokens_accepted
            
                empirical_alpha = total_generated_tokens / ((self.max_proposals_per_step + 1) * propose_steps) # accepted_tokens_per_iter / ((k+1) * num_cycles)
                rewards.append(empirical_alpha)
        return {"reward": rewards, "probability": torch.prod(torch.tensor(probs))}
        
class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, hidden_states):
        return self.value_head(hidden_states).squeeze(-1)
    

class PPOConfig:
    def __init__(
        self,
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        max_epochs: int = 100,
        clip_range: float = 0.2,
        value_clip_range: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        checkpoint_interval: int = 10
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.checkpoint_interval = checkpoint_interval

class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, hidden_states):
        return self.value_head(hidden_states).squeeze(-1)

class SpeculativePPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        draft_model: nn.Module,
        target_model: nn.Module,
        tokenizer,
        dataset,
        device: str = "cuda"
    ):
        self.config = config
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

        self.num_rollouts = 5
        
        # Add value heads to both models
        self.draft_value_head = ValueHead(draft_model.config.hidden_size).to(device)
        self.target_value_head = ValueHead(target_model.config.hidden_size).to(device)
        
        # Optimizers
        print(config.learning_rate)
        self.policy_optimizer = optim.Adam(
            list(self.draft_model.parameters()) + list(self.draft_value_head.parameters()),
            lr=config.learning_rate
        )
        
        self.spec_decoder = SpeculativeDecoder(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            device=device,
            num_rollouts=self.num_rollouts
        )
        
        self.step_count = 0

    def compute_gae(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            
        returns = advantages + torch.tensor(values).to(advantages.device)
        return advantages, returns

    def compute_draft_model_probs(self, draft_model, draft_value_head, input_ids, num_rollouts, max_proposals_per_step):
        new_log_probs = torch.tensor([]).cuda()
        values = torch.tensor([]).cuda()
        draft_model.train()
        draft_value_head.train()
        for _ in range(num_rollouts):
            scores = torch.tensor([]).cuda()
            for i in range(max_proposals_per_step):
                output = draft_model(input_ids.squeeze(0), output_hidden_states=True, return_dict=True)
                if len(scores) == 0:
                    scores = output.logits[0][-1].unsqueeze(0)
                else:
                    scores = torch.cat([scores, output.logits[0][-1].unsqueeze(0)])
                    input_ids = torch.cat([input_ids, torch.multinomial(F.softmax(output.logits[0][-1]), num_samples=1).unsqueeze(dim=0).unsqueeze(dim=0)], dim=-1)

            if len(values) == 0:
                values = draft_value_head(output.hidden_states[-1][-1][-1].float()).unsqueeze(0)
            else:
                values = torch.cat([values, draft_value_head(output.hidden_states[-1][-1][-1].float()).unsqueeze(0)])
            draft_logits = scores
            draft_probs = F.softmax(draft_logits, dim=-1)
            
            # Sample proposals
            
            proposed_tokens = torch.multinomial(draft_probs, num_samples=1)
            probs = draft_probs[i, proposed_tokens[i]]
            for i in range(1, len(proposed_tokens)):
                probs = torch.cat([probs, draft_probs[i, proposed_tokens[i]]])
            if len(new_log_probs) == 0:
                new_log_probs = torch.prod(probs).unsqueeze(0)
            else:
                new_log_probs = torch.cat([new_log_probs, torch.prod(probs).unsqueeze(0)])
        return new_log_probs, values

    def train_step(
        self,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Execute single PPO training step"""
        new_log_probs, new_values = self.compute_draft_model_probs(self.draft_model, self.draft_value_head, input_ids, self.num_rollouts, self.spec_decoder.max_proposals_per_step)
        # Calculate policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(
            ratio,
            1 - self.config.clip_range,
            1 + self.config.clip_range
        ) * advantages
        
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Calculate value loss
        value_pred_clipped = values + torch.clamp(
            new_values - values,
            -self.config.value_clip_range,
            self.config.value_clip_range
        )
        value_loss1 = (new_values - returns) ** 2
        value_loss2 = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        # Calculate entropy loss
        entropy = -(torch.exp(new_log_probs) * new_log_probs).sum(dim=-1).mean()

        # Total loss
        total_loss = (
            policy_loss
            - self.config.entropy_coef * entropy
            + self.config.value_loss_coef * value_loss
        )
        
        # Optimize
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.draft_model.parameters()) + list(self.draft_value_head.parameters()),
            self.config.max_grad_norm
        )
        self.policy_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "approx_kl": ((old_log_probs - new_log_probs) * torch.exp(old_log_probs)).sum(dim=-1).mean().item()
        }

    def train(self):
        """Execute full training loop"""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.config.max_epochs):
            epoch_stats = defaultdict(list)
            
            for batch in tqdm(dataloader):
                # Generate responses and compute rewards using speculative decoding
                spec_outputs = [
                    self.spec_decoder.generate(input_ids.to(self.device))
                    for input_ids in batch["input_ids"]
                ]
                # Collect rewards and other metrics
                rewards = torch.tensor([
                    output['reward'] for output in spec_outputs
                ], device=self.device)
                
                # Get old policy distribution and values
                _, old_values = self.compute_draft_model_probs(self.draft_model, self.draft_value_head, batch["input_ids"].to(self.device), self.num_rollouts, self.spec_decoder.max_proposals_per_step)

                    
                # Compute advantages and returns
                advantages, returns = self.compute_gae(
                    old_values,
                    rewards,
                    torch.zeros_like(rewards),  # dones
                    old_values  # next_values (using same values as simple approximation)
                )
                
                # Execute training step
                stats = self.train_step(
                    torch.stack([output['probability'] for output in spec_outputs]).cuda(),
                    old_values,
                    advantages,
                    returns,
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device)
                )
                
                # Log stats
                for key, value in stats.items():
                    epoch_stats[key].append(value)
                
                self.step_count += 1
                
                # Log to wandb
                wandb.log({
                    "global_step": self.step_count,
                    **{f"batch/{k}": v for k, v in stats.items()},
                    "batch/mean_reward": rewards.mean().item(),
                    "batch/mean_acceptance_rate": np.mean([
                        output['reward'] for output in spec_outputs
                    ])
                })
            
            # Log epoch summary
            wandb.log({
                "epoch": epoch,
                **{f"epoch/{k}": np.mean(v) for k, v in epoch_stats.items()}
            })
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoints/spec-ppo-epoch-{epoch+1}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            "draft_model": self.draft_model.state_dict(),
            "draft_value_head": self.draft_value_head.state_dict(),
            "optimizer": self.policy_optimizer.state_dict(),
            "step_count": self.step_count,
            "config": self.config.__dict__
        }, path)
        wandb.save(path)

def init_wandb(config: PPOConfig):
    """Initialize Weights & Biases project"""
    run_name = f"spec-ppo-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="speculative-ppo",
        name=run_name,
        config=config.__dict__
    )


def setup_logging(log_dir):
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training.log"),
            logging.StreamHandler()
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Train Speculative PPO')
    
    # Model arguments
    parser.add_argument('--draft_model_path', type=str, required=True,
                      help='Path to draft model')
    parser.add_argument('--target_model_path', type=str, required=True,
                      help='Path to target model')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='HuggingFace dataset name')
    parser.add_argument('--dataset_config', type=str, default=None,
                      help='Dataset configuration name')
    parser.add_argument('--split', type=str, default='train',
                      help='Dataset split to use')
    parser.add_argument('--text_column', type=str, required=True,
                      help='Column name containing the text data')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--max_proposals_per_step', type=int, default=4)
    
    # PPO specific arguments
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--value_clip_range', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    
    # Other arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save checkpoints and logs')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', type=str, default='speculative-ppo')
    parser.add_argument('--wandb_entity', type=str, default=None)
    
    return parser.parse_args()

def prepare_dataset(dataset, tokenizer, text_column, max_length):
    """Prepare dataset for training"""
    
    def tokenize_function(example):
        # Apply chat template if it's conversation data
        texts = tokenizer.apply_chat_template([{"role": "user", "content": example[text_column]}], tokenize=False, add_generation_prompt=True)
        # Tokenize
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        example.update({
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask
        })
        return example
    
    # Apply tokenization to dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    return tokenized_dataset

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup logging
    setup_logging(args.output_dir)
    logging.info(f"Arguments: {args}")
    
    # Initialize wandb
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    try:
        # Load tokenizer
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load models
        logging.info("Loading models...")
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load and prepare dataset
        logging.info("Loading dataset...")
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.split
        )
        
        logging.info("Preparing dataset...")
        processed_dataset = prepare_dataset(
            dataset,
            tokenizer,
            args.text_column,
            args.max_length
        )
        
        # Initialize PPO config
        config = PPOConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            clip_range=args.clip_range,
            value_clip_range=args.value_clip_range,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = SpeculativePPOTrainer(
            config=config,
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            dataset=processed_dataset,
            device=device
        )
        
        # Configure speculative decoder settings
        trainer.spec_decoder.max_new_tokens = args.max_new_tokens
        trainer.spec_decoder.max_proposals_per_step = args.max_proposals_per_step
        
        # Start training
        logging.info("Starting training...")
        trainer.train()
        
        # Save final model
        logging.info("Saving final model...")
        final_output_dir = os.path.join(args.output_dir, "final_model")
        trainer.save_checkpoint(final_output_dir)
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Close wandb run
        wandb.finish()
        logging.info("Training completed")

if __name__ == "__main__":
    main()