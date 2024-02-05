from .base import BaseAWQForCausalLM
from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertLayer

class BertAWQModel(BaseAWQForCausalLM):
    layer_type = "BertEncoder"
    max_new_tokens_key = "n_positions"

    @staticmethod
    def get_model_layers(model: BertModel):
        def prepare_inputs_for_generation(input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}
        model.prepare_inputs_for_generation = prepare_inputs_for_generation
        return model.encoder.layer
    
    @staticmethod
    def get_act_for_scaling(module: BertEncoder):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.fc_in.out_features
        )
    
    
    def move_embed(self, model: BertModel, device: str):
        model.embeddings = model.embeddings.to(device)
                
        if model.pooler is not None:
            model.pooler.dense = model.pooler.dense.to(device)
    
    def get_layers_for_scaling(self, module: BertLayer, input_feat, module_kwargs):
        layers = []
        
        # module.attention
        # TODO: Handle NoOp. No previous LayerNorm/Linear in module.attention like in other models.
        # layers.append(dict(
        #     prev_op=module.identity, 
        #     layers=[module.attention.self.query,
        #             module.attention.self.key, module.attention.self.value],
        #     inp=input_feat['attention.self.query'],
        #     module2inspect=module.attention, kwargs=module_kwargs,
        # ))

        # attention out
        layers.append(dict(
            prev_op=module.attention.self.value,
            layers=[module.attention.output.dense],
            inp=input_feat['attention.self.value'],
        ))

        # # linear 2
        # layers.append(dict(
        #     prev_op=module.intermediate.intermediate_act_fn,
        #     layers=[module.output.dense],
        #     inp=input_feat['intermediate.dense'],
        # ))
        
        # # linear 1
        # layers.append(dict(
        #     prev_op=module.attention.output.dropout,
        #     layers=[module.intermediate.dense],
        #     inp=input_feat['attention.output.dropout'],
        # ))        

        return layers
    
    