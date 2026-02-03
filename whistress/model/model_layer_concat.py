from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    PreTrainedModel,
    WhisperConfig,
)
from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class CustomModelOutput(BaseModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    head_preds: torch.FloatTensor = None
    labels_head: Optional[torch.FloatTensor] = None
    whisper_logits: torch.FloatTensor = None
    preds: Optional[torch.Tensor] = None


class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNN, self).__init__()
        hidden_dim = 2 * input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- NEW: MLP Projector for fusing Encoder Layers ---
class EncoderLayerProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Input is 2 * d_model (Layer 1 + Layer 9)
        # Output is d_model (to feed into Decoder Block)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU() 

    def forward(self, x):
        # x shape: [Batch, Seq, Dim * 2]
        return self.activation(self.linear(x))


class WhiStress(PreTrainedModel):

    config_class = WhisperConfig
    model_input_names = ["input_features", "labels_head", "whisper_labels"]

    def __init__(
        self,
        config: WhisperConfig,
        layer_for_head: Optional[int] = None, 
        whisper_backbone_name="openai/whisper-small.en",
    ):
        super().__init__(config)
        self.whisper_backbone_name = whisper_backbone_name
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.whisper_backbone_name,
        ).eval()
        self.processor = WhisperProcessor.from_pretrained(self.whisper_backbone_name)

        input_dim = self.whisper_model.config.d_model 
        output_dim = 2 

        config = self.whisper_model.config
        
        # --- ARCHITECTURAL CHANGE: Encoder Fusion ---
        # 1. We keep the original decoder block logic intact
        self.additional_decoder_block = WhisperDecoderLayer(config)
        
        # 2. We add a projector to fuse Layer 1 + Layer 9 down to one vector
        # Input: d_model (Layer 1) + d_model (Layer 9) = 2 * d_model
        # Output: d_model
        self.encoder_projector = EncoderLayerProjector(input_dim * 2, input_dim)
        
        self.classifier = FCNN(input_dim, output_dim)
        
        neg_weight = 1.0
        pos_weight = 0.7 / 0.3
        class_weights = torch.tensor([neg_weight, pos_weight])
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
        
        # Default Layer 9 (for Text and for semantic Audio)
        self.layer_for_head = 9 if layer_for_head is None else layer_for_head
        
        # The Acoustic Layer we want to inject (Layer 1 has high fidelity)
        self.acoustic_layer_idx = 3

    def to(self, device: str = ("cuda" if torch.cuda.is_available() else "cpu")):
        self.whisper_model.to(device)
        self.additional_decoder_block.to(device)
        self.encoder_projector.to(device)
        self.classifier.to(device)
        super().to(device)
        return self

    def load_model(self, save_dir=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        if save_dir is not None:
            print('loading model from:', save_dir)
            self.classifier.load_state_dict(
                torch.load(os.path.join(save_dir, "classifier.pt"), map_location=device, weights_only=False)
            )
            self.additional_decoder_block.load_state_dict(
                torch.load(os.path.join(save_dir, "additional_decoder_block.pt"), map_location=device, weights_only=False)
            )
            
            # Load the Projector (New Architecture)
            # You must train a new model to generate this file.
            proj_path = os.path.join(save_dir, "encoder_projector.pt")
            if os.path.exists(proj_path):
                self.encoder_projector.load_state_dict(
                    torch.load(proj_path, map_location=device, weights_only=False)
                )

            # Legacy metadata support
            meta_path = os.path.join(save_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                    self.layer_for_head = metadata.get("layer_for_head", 9)

    def train(self, mode: Optional[bool] = True):
        self.whisper_model.eval()
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        
        for param in self.additional_decoder_block.parameters():
            param.requires_grad = True
        for param in self.encoder_projector.parameters(): # Train the MLP
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        self.additional_decoder_block.train()
        self.encoder_projector.train()
        self.classifier.train()

    def eval(self):
        self.whisper_model.eval()
        self.additional_decoder_block.eval()
        self.encoder_projector.eval()
        self.classifier.eval()

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels_head=None,
        whisper_labels=None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model.eval()

        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            labels=whisper_labels,
        )

        # 1. Decoder Hidden State (Query) - Layer 9
        # "Where are we in the text?"
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[self.layer_for_head].to(device)

        # --- FUSION STEP ---
        # 2a. Encoder Layer 9 ("General Prosody/Meaning")
        encoder_semantic = backbone_outputs.encoder_hidden_states[self.layer_for_head].to(device)
        
        # 2b. Encoder Layer 1 ("Raw Acoustics/Pitch")
        encoder_acoustic = backbone_outputs.encoder_hidden_states[self.acoustic_layer_idx].to(device)
        
        # 3. Concatenate Features: [Batch, Audio_Seq, 768 * 2]
        raw_fused_encoder = torch.cat([encoder_acoustic, encoder_semantic], dim=-1)
        
        # 4. Project back to d_model: [Batch, Audio_Seq, 768]
        # This mixes the acoustic detail with semantic context BEFORE the decoder block
        fused_encoder_states = self.encoder_projector(raw_fused_encoder)
        
        # -------------------

        # 5. Pass to Decoder Block (Cross-Attention)
        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states, # Text Query
            encoder_hidden_states=fused_encoder_states      # Fused Audio Key/Value
        )
        
        head_logits = self.classifier(additional_decoder_block_outputs[0].to(device))

        # ... (standard output calculation) ...
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        if labels_head is not None:
            preds = torch.where(
                torch.isin(
                    labels_head, torch.tensor(list([-100])).to(device)
                ),
                torch.tensor(-100),
                preds,
            )
        
        loss = None
        if labels_head is not None:
            loss = self.loss_fct(
                head_logits.reshape(-1, head_logits.size(-1)), labels_head.reshape(-1)
            )
            
        return CustomModelOutput(
            logits=head_logits,
            labels_head=labels_head,
            whisper_logits=backbone_outputs.logits,
            loss=loss,
            preds=preds,
        )

    def generate(
        self,
        input_features,
        max_length=128,
        labels_head=None,
        whisper_labels=None,
        **generate_kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_outputs = self.whisper_model.generate(
            input_features=input_features,
            max_length=max_length,
            labels=whisper_labels,
            do_sample=False,
            **generate_kwargs,
        )

        backbone_outputs = self.whisper_model(
            input_features=input_features,
            decoder_input_ids=whisper_outputs,
            output_hidden_states=True,
        )

        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[self.layer_for_head].to(device)

        # Replicate Fusion Logic
        encoder_semantic = backbone_outputs.encoder_hidden_states[self.layer_for_head].to(device)
        encoder_acoustic = backbone_outputs.encoder_hidden_states[self.acoustic_layer_idx].to(device)
        
        raw_fused_encoder = torch.cat([encoder_acoustic, encoder_semantic], dim=-1)
        fused_encoder_states = self.encoder_projector(raw_fused_encoder)

        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=fused_encoder_states,
        )
        
        head_logits = self.classifier(additional_decoder_block_outputs[0].to(device))
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        preds = torch.where(
            torch.isin(
                whisper_outputs, torch.tensor(list([50256])).to(device)
            ),
            torch.tensor(-100),
            preds,
        )
        return preds

    def generate_dual(
        self,
        input_features,
        attention_mask=None,
        max_length=200,
        labels_head=None,
        whisper_labels=None,
        **generate_kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_outputs = self.whisper_model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_length=max_length,
            labels=whisper_labels,
            return_dict_in_generate=True,
            **generate_kwargs,
        )

        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=whisper_outputs.sequences,
            output_hidden_states=True,
        )

        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[self.layer_for_head].to(device)

        # Replicate Fusion Logic
        encoder_semantic = backbone_outputs.encoder_hidden_states[self.layer_for_head].to(device)
        encoder_acoustic = backbone_outputs.encoder_hidden_states[self.acoustic_layer_idx].to(device)
        
        raw_fused_encoder = torch.cat([encoder_acoustic, encoder_semantic], dim=-1)
        fused_encoder_states = self.encoder_projector(raw_fused_encoder)

        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=fused_encoder_states,
        )
        
        head_logits = self.classifier(additional_decoder_block_outputs[0].to(device))
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        preds = torch.where(
            torch.isin(
                whisper_outputs.sequences, torch.tensor(list([50256])).to(device)
            ),
            torch.tensor(-100),
            preds,
        )
        return CustomModelOutput(
            logits=head_logits,
            head_preds=preds,
            whisper_logits=whisper_outputs.logits,
            preds=whisper_outputs.sequences
        )

    def __str__(self):
        return "WhiStress"