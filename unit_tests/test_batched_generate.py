from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
import unittest
import torch
import gc

quant_path = "casperhansen/vicuna-7b-v1.5-awq"
quant_file = "awq_model_w4_g128.pt"

class TestBatched(unittest.TestCase):
    def setUp(self):
        # Load model
        self.model = AutoAWQForCausalLM.from_quantized(quant_path, quant_file, fuse_layers=True)
        self.tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True, padding_side='left')
        self.streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
        
        # Convert prompt to tokens
        self.prompt_template = """\
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {prompt}
ASSISTANT:"""


    def test_single_prompt(self):

        prompt = self.prompt_template.format(prompt="How are you today?")

        tokens = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).input_ids.cuda()
        
        # Generate output
        generation_output = self.model.generate(
            tokens, 
            max_new_tokens=100
        )
        # test we received exactly one response
        self.assertEqual(generation_output.shape[0], 1)
        # test response is not empty
        self.assertTrue(generation_output.shape[1] > 0)
        try:
            del self.model
            del self.tokenizer
        except:
            pass
        _ = gc.collect()
        torch.cuda.empty_cache()

    def test_batch_size_1(self):

        prompt_batch = [self.prompt_template.format(prompt="How are you today?"), 
                        ]

        tokens = self.tokenizer(
            prompt_batch,
            padding='longest',
            return_tensors='pt'
        ).input_ids.cuda()
        
        # Generate output
        generation_output = self.model.generate(
            tokens, 
            max_new_tokens=512
        )
        # test we received exactly one response
        self.assertEqual(generation_output.shape[0], 1)
        # test response is not empty
        self.assertTrue(generation_output.shape[1] > 0)
        try:
            del self.model
            del self.tokenizer
        except:
            pass
        _ = gc.collect()
        torch.cuda.empty_cache()

    def test_batch_size_2(self):

        prompt_batch = [self.prompt_template.format(prompt="How are you today?"), 
                        self.prompt_template.format(prompt="In what country is Cairo located?"), 
                        ]

        tokens = self.tokenizer(
            prompt_batch,
            padding='longest',
            return_tensors='pt'
        ).input_ids.cuda()
        
        # Generate output
        generation_output = self.model.generate(
            tokens, 
            max_new_tokens=512
        )
        # test we got two outputs, one for each prompt
        self.assertEqual(generation_output.shape[0], 2)
        # test output is not empty
        self.assertTrue(generation_output.shape[1] > 0)
        # make sure we are getting different answers for prompts in batch
        self.assertTrue(generation_output[0][1] != generation_output[1][1])
        print(generation_output.shape)
        try:
            del self.model
            del self.tokenizer
        except:
            pass
        _ = gc.collect()
        torch.cuda.empty_cache()

    def test_batch_size_4(self):

        prompt_batch = [self.prompt_template.format(prompt="How are you today?"), 
                        self.prompt_template.format(prompt="In what country is Cairo located?"), 
                        self.prompt_template.format(prompt="How tall is Mt. Everest?"), 
                        self.prompt_template.format(prompt="What is your favorite ice cream flavor?"), 
                        ]

        tokens = self.tokenizer(
            prompt_batch,
            padding='longest',
            return_tensors='pt'
        ).input_ids.cuda()
        
        # Generate output
        generation_output = self.model.generate(
            tokens, 
            max_new_tokens=512
        )
        # test we got 4 outputs, one for each prompt
        self.assertEqual(generation_output.shape[0], 4)
        # test output is not empty
        self.assertTrue(generation_output.shape[1] > 0)
        try:
            del self.model
            del self.tokenizer
        except:
            pass
        _ = gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()
