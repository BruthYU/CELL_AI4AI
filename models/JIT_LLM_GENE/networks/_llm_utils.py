import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel, PreTrainedModel
class NoRoPE(nn.Module):
    """
    A drop-in replacement for LlamaRotaryEmbedding that always returns:
      cos = all ones, sin = all zeros
    of shape (batch_size, seq_len, head_dim), so rotary has no effect.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _hidden_dim = hidden_states.shape

        # Create cos = ones, sin = zeros
        #   shape --> (batch_size, seq_len, head_dim)
        cos = hidden_states.new_ones(batch_size, seq_len, self.head_dim)
        sin = hidden_states.new_zeros(batch_size, seq_len, self.head_dim)
        return cos, sin
        
class LlamaBidirectionalModel(LlamaModel):
    """
    A drop-in replacement for LlamaModel with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.rotary_emb = NoRoPE(
            head_dim=config.head_dim,
        )

        # Explicitly disable causal attention
        self.config.is_causal = False
        # force every layer to be non-causal
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal‐(look‐ahead) masking.
        # The only mask that remains is whatever "attention_mask" the user has passed
        # (e.g. padding‐mask), which will be handled by Flash/SDPA internally as non‐causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **flash_attn_kwargs,
    ):
        flash_attn_kwargs["is_causal"] = False

        # If no attention_mask is provided, create an all-ones mask (no masking)
        # This ensures bidirectional attention with correct device/dtype
        if attention_mask is None:
            # Get batch size (B) and sequence length (S) from input_embeds if available, else from input_ids.
            # If neither is available, fall back to attention_mask=None and log a warning.
            B = None
            S = None
            if inputs_embeds is not None:
                B, S = inputs_embeds.size(0), inputs_embeds.size(1)
            if B and S:
                attention_mask = torch.ones((B, 1, S, S), dtype=torch.float, device=inputs_embeds.device)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

class GPT2BidirectionalModel(GPT2Model):
    """
    A thin wrapper around GPT2Model that disables the causal (unidirectional) mask,
    allowing full bidirectional attention—and prints the internal bias mask each forward pass.
    """

    def __init__(self, config: GPT2Config):
        # Mark as not‐a‐decoder (for downstream utilities).
        config.is_decoder = False
        super().__init__(config)

        # Overwrite each attention's bias so no triangular masking occurs.
        for block in self.h:
            # block.attn.bias is a bool‐tensor of shape (1, 1, max_pos, max_pos).
            block.attn.bias.data.fill_(True)
            block.attn.is_causal = False

        def _no_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            return None

        self._update_causal_mask = _no_causal_mask.__get__(self, GPT2Model)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Determine sequence length for printing the relevant slice of bias
        if input_ids is not None:
            seq_len = input_ids.size(1)
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.size(1)
        else:
            seq_len = None  # If neither is given, we can’t infer seq_len

        if seq_len is not None:
            # Print the (1, 1, seq_len, seq_len) slice of the bias for the first block
            bias_mask = self.h[0].attn.bias[0, 0, :seq_len, :seq_len]
        #     print("Bias mask (block 0) slice [0,0,:seq_len,:seq_len]:")
        #     print(bias_mask)
        # else:
        #     print("Cannot infer sequence length to print bias mask.")

        # If a 2D attention_mask was provided, print its expanded 4D version:
        if attention_mask is not None:
            # Expand to (batch_size, 1, seq_len, seq_len)
            B, S = attention_mask.size()
            expanded = attention_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, S, S)
            # Convert to float mask (1→0.0, 0→-inf) just like GPT2 does internally
            neg_inf = torch.finfo(self.dtype).min
            float_mask = (1.0 - expanded.to(self.dtype)) * neg_inf
            # print(f"Expanded attention_mask (shape {expanded.shape}) → float mask:")
            # print(float_mask)

        # Finally, call the parent forward method
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


def get_transformer_backbone(key, kwargs) -> PreTrainedModel:
    kwargs = dict(kwargs or {})

    if key == "GPT2":
        config = GPT2Config(**kwargs)
        model = GPT2BidirectionalModel(config)

        # Zero out position embeddings and freeze them
        model.wpe.weight.requires_grad = False
        model.wte.weight.requires_grad = False
        model.wpe.weight.zero_()
        model.wte.weight.zero_()

        model_dim = config.n_embd
    elif key == "llama":
        bidirectional_attention = bool(kwargs.pop("bidirectional_attention", False))

        config = LlamaConfig(**kwargs)
        if bidirectional_attention:
            model = LlamaBidirectionalModel(config)
        else:
            model = LlamaModel(config)
        model_dim = config.hidden_size

        model.embed_tokens.weight.requires_grad = False
        model.embed_tokens.weight.zero_()
    else:
        raise ValueError(f"Unknown backbone key {key}")

    return model, model_dim

if __name__ == "__main__":
    kwargs = {
        "vocab_size": 100,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
    }
    model, model_dim = get_transformer_backbone("GPT2", kwargs)
    # model, model_dim = get_transformer_backbone("llama", kwargs)
    model.to("cuda")
    model.eval()
    batch_size, seq_length, hidden_dim = (16, 128, 512)
    seq_input = torch.randn(batch_size, seq_length, hidden_dim, device=model.device)

    outputs = model(inputs_embeds=seq_input)
    print(outputs.keys())
    pass


