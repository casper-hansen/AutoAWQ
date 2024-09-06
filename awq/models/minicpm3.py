
from .base import BaseAWQForCausalLM
from ..quantize.quantizer import AwqQuantizer
import torch
from .base import (
     Annotated,
     AwqConfig,
     BaseAWQForCausalLM,
     Dict,
     Doc,
     List,
     PreTrainedTokenizer,
     Union,
 )
from ..quantize.quantizer import AwqQuantizer, clear_memory, get_best_device

class CPM3AwqQuantizer(AwqQuantizer):
    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        if org_w_shape[0] % oc_batch_size != 0:
            oc_batch_size = org_w_shape[0]
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

class MiniCPM3AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MiniCPMDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    @torch.no_grad()
    def quantize(
        self,
        tokenizer: Annotated[
            PreTrainedTokenizer, Doc("The tokenizer to use for quantization.")
        ] = None,
        quant_config: Annotated[
            Dict, Doc("The quantization config you want to use.")
        ] = {},
        calib_data: Annotated[
            Union[str, List[str]],
            Doc(
                "The calibration dataset. Either a string pointing to Huggingface or a list of preloaded examples."
            ),
        ] = "pileval",
        split: Annotated[str, Doc("The split of calib_data.")] = "train",
        text_column: Annotated[str, Doc("The text column of calib_data.")] = "text",
        duo_scaling: Annotated[
            bool, Doc("Whether to scale using both w/x or just x.")
        ] = True,
        export_compatible: Annotated[
            bool,
            Doc(
                "This argument avoids real quantization by only applying the scales without quantizing down to FP16."
            ),
        ] = False,
        apply_clip: Annotated[
            bool,
            Doc(
                "Whether to apply clipping to the model during quantization. Some models may perform better with this set to False."
            ),
        ] = True,
        n_parallel_calib_samples: Annotated[
            int,
            Doc(
                "The number of parallel samples to run through the model. "
                "A high number of parallel samples can result in OOM during quantization if max_calib_samples is high enough. "
                "If None, runs through all samples at the same time. "
                "You can set this to a low number for more memory efficient quantization."
            ),
        ] = None,
        max_calib_samples: Annotated[
            int, Doc("The maximum number of samples to run through the model.")
        ] = 128,
        max_calib_seq_len: Annotated[
            int,
            Doc(
                "The maximum sequence length of the calibration dataset. Discard samples greater than max_calib_seq_len."
            ),
        ] = 512,
        max_chunk_memory: Annotated[
            int,
            Doc(
                "The loss computation and per-channel mean is optimized into chunked computations."
                " Adjust this parameter to increase or decrease memory usage for these computations."
                " Default is 1GB (1024 * 1024 * 1024)."
            ),
        ] = 1024
        * 1024
        * 1024,
    ):
        """
        The main quantization function that you can use to quantize your model.

        Example:

        ```python
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model_path = "..."
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        model.quantize(tokenizer, quant_config)
        ```
        """
        self.quant_config: AwqConfig = AwqConfig.from_dict(quant_config)

        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert

        self.quantizer = CPM3AwqQuantizer(
            self,
            self.model,
            tokenizer,
            self.quant_config.w_bit,
            self.quant_config.q_group_size,
            self.quant_config.zero_point,
            self.quant_config.version,
            calib_data,
            split,
            text_column,
            duo_scaling,
            modules_to_not_convert=self.quant_config.modules_to_not_convert,
            export_compatible=export_compatible,
            apply_clip=apply_clip,
            n_parallel_calib_samples=n_parallel_calib_samples,
            max_calib_samples=max_calib_samples,
            max_calib_seq_len=max_calib_seq_len,
            max_chunk_memory=max_chunk_memory,
        )
        self.quantizer.quantize()

        self.is_quantized = True
    @staticmethod
    def get_model_layers(model):
        print(model.model.layers)
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # layers.append(
        #     dict(
        #         prev_op=module.input_layernorm,
        #         layers=[
        #             module.self_attn.q_a_proj,
        #         ],
        #         inp=input_feat["self_attn.q_a_proj"],
        #         module2inspect=module.self_attn.q_a_proj,
        #         kwargs=module_kwargs,
        #     )
        # )
        # mlp
        layers.append(
            dict(
                prev_op=module.self_attn.q_a_layernorm,
                layers=[
                    module.self_attn.q_b_proj,
                    
                ],
                inp=input_feat["self_attn.q_b_proj"],
                module2inspect=module.self_attn.q_b_proj,
                kwargs=module_kwargs,
            )
        )

        layers.append(
            dict(
                prev_op=module.self_attn.kv_a_layernorm,
                layers=[
                    module.self_attn.kv_b_proj,
                ],
                inp=input_feat["self_attn.kv_b_proj"],
                module2inspect=module.self_attn.kv_b_proj,
                kwargs=module_kwargs,
            )
        )
        

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj,module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp
            )
        )

        return layers


