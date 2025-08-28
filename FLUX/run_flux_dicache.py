from typing import Any, Dict, Optional, Tuple, Union
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def dicache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        if self.enable_dicache:
            # ------------------ Online Probe Profiling Scheme --------------------
            if self.cnt <= int(self.ret_ratio * self.num_steps) or self.cnt == self.num_steps - 1: # direct inference, do not need to resume from the probe state
                should_calc = True
                self.resume_flag = False
                self.accumulated_rel_l1_distance = 0
            else: # probe first, if able to use cache, resume from the probe state; else skip this step
                test_hidden_states, test_encoder_hidden_states = hidden_states.clone(), encoder_hidden_states.clone()
                probe_blocks = self.transformer_blocks[0:self.probe_depth]
                for probe_block in probe_blocks:
                    test_encoder_hidden_states, test_hidden_states = probe_block(
                        hidden_states=test_hidden_states,
                        encoder_hidden_states=test_encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                
                delta_x = (hidden_states - self.previous_input).abs().mean() / self.previous_input.abs().mean()
                delta_y = (test_hidden_states - self.previous_probe_states).abs().mean() / self.previous_probe_states.abs().mean()

                if self.error_choice =="delta_minus":
                    error = (delta_y - delta_x).abs()
                elif self.error_choice == "delta_y":
                    error = delta_y
                
                self.accumulated_rel_l1_distance += error

                if self.accumulated_rel_l1_distance < self.rel_l1_thresh: # skip this step
                    should_calc = False 
                    self.resume_flag = False
                else: # continue calculating, but can resume from the probe state
                    should_calc = True
                    self.resume_flag = True
                    self.accumulated_rel_l1_distance = 0
            # --------------------------------------------------------------------

        if self.enable_dicache:
            if not should_calc:
                
                ori_hidden_states = hidden_states.clone()

                # ------------------ Dynamic Cache Trajectory Alignment --------------------
                if len(self.residual_window) >= 2:
                    current_residual_indicator = test_hidden_states - hidden_states
                    gamma = ((current_residual_indicator - self.probe_residual_window[-2]).abs().mean() / (self.probe_residual_window[-1] - self.probe_residual_window[-2]).abs().mean()).clip(1, 1.5)
                    hidden_states += self.residual_window[-2] + gamma * (self.residual_window[-1] - self.residual_window[-2])
                else:
                    hidden_states += self.previous_residual 
                # ---------------------------------------------------------------------------

                self.previous_probe_states = test_hidden_states
                self.previous_input = ori_hidden_states


            else:
                # -------------- resume from previously calculated result --------------
                ori_hidden_states = hidden_states.clone()
                if self.resume_flag: # resume from the probe state
                    hidden_states = test_hidden_states
                    encoder_hidden_states = test_encoder_hidden_states
                    unpass_transformer_blocks = self.transformer_blocks[self.probe_depth:]
                else:
                    unpass_transformer_blocks = self.transformer_blocks
                # ----------------------------------------------------------------------
                
                for index_block, block in enumerate(unpass_transformer_blocks): # self.transformer_blocks
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                    
                    # --------------- record probe feature ------------------
                    if index_block == self.probe_depth - 1:
                        if self.cnt <= int(self.ret_ratio * self.num_steps) or self.cnt == self.num_steps - 1:
                            self.previous_probe_states = hidden_states
                        else:
                            self.previous_probe_states = test_hidden_states
                    # --------------------------------------------------------
                    
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                for index_block, block in enumerate(self.single_transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        hidden_states = block(
                            hidden_states=hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )

                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                
                self.previous_residual = hidden_states - ori_hidden_states # residual between block 0 and block M 
                self.previous_probe_residual = self.previous_probe_states - ori_hidden_states # residual between block 0 and block m 
                self.previous_input = ori_hidden_states
                self.previous_output = hidden_states

                self.residual_window.append(self.previous_residual)
                self.probe_residual_window.append(self.previous_probe_residual)
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        self.cnt += 1 
        if self.cnt == self.num_steps:
            self.cnt = 0        
            self.residual_window = []
            self.probe_residual_window = []

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

FluxTransformer2DModel.forward = dicache_forward
pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
num_inference_steps = 30
seed = 0

prompt = "a dog plays on the grass."

# TeaCache
pipeline.transformer.__class__.enable_dicache = True
pipeline.transformer.__class__.cnt = 0
pipeline.transformer.__class__.num_steps = num_inference_steps
pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
pipeline.transformer.__class__.previous_modulated_input = None
pipeline.transformer.__class__.previous_residual = None
pipeline.transformer.__class__.residual_window = []
pipeline.transformer.__class__.probe_residual_window = []
pipeline.transformer.__class__.probe_depth = 1 # recommend 1~5

error_choice = "delta_y" # choose from ["delta_minus", "delta_y"]

if error_choice == "delta_minus": # use (dy - dx) to simulate caching error
    pipeline.transformer.__class__.error_choice = "delta_minus"
    pipeline.transformer.__class__.rel_l1_thresh = 0.08
    pipeline.transformer.__class__.ret_ratio = 0.0
elif error_choice == "delta_y": # use dy to simulate caching error
    pipeline.transformer.__class__.error_choice = "delta_y"
    pipeline.transformer.__class__.rel_l1_thresh = 0.4
    pipeline.transformer.__class__.ret_ratio = 0.2


pipeline.to("cuda")
img = pipeline(
    prompt, 
    num_inference_steps=num_inference_steps,
    generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]

img.save("{}.png".format('DiCache_' + prompt))

