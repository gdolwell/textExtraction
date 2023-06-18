import argparse
import numpy as np
import math
import zlib
import pandas as pd
from llm import LLM
from transformers.generation_logits_process import LogitsProcessor, LogitsProcessorList
import torch

'''Named Huggingface models are assumed to be located at ~/.cache/huggingface/hub'''

def do_zlib(text:str):
    return np.log(len(zlib.compress(bytes(text, 'utf-8'))))

class DecayingTemperatureWarper(LogitsProcessor):
    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature
        self.mapping = {1: 10.0, 2: 9.53, 3: 9.06, 4: 8.59, 5: 8.12, 6: 7.65, 7: 7.18, 8: 6.71, 9: 6.24, 10: 5.77, 11: 5.30, 
                        12: 4.83, 13: 4.36, 14: 3.89, 15: 3.42, 16: 2.95, 17: 2.49, 18: 2.01, 19: 1.54, 20: 1.0}

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        self.temperature = self.mapping.get(cur_len, 1.0)
        
        return scores
    
def parse_commoncrawl(wetfile):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wetfile) as f:
        lines = f.readlines() 
    
    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]

    return all_eng

def main(args):
    print("Loading models...")
    target_model= LLM(args.target)

    # number of tokens to generate and top k sampling tuning parameters
    seq_len = 256
    top_k = 40

    num_batches = int(math.ceil(args.n / args.batchsize))
    out_df = pd.DataFrame()
    out_df['text'] = []

    # Batched sequence generation, gerenates padding warning see: 
    # https://github.com/huggingface/transformers/blob/118e9810687dd713b6be07af79e80eeb1d916908/src/transformers/generation/utils.py#L1314-L1317
    if args.dotemp:
        print('Temperature decay sampling')
        logits_warper = LogitsProcessorList([DecayingTemperatureWarper(10.0)])

        for batch in range(num_batches):
            print(f"\nBatch number: {batch} of {num_batches}.\n")

            prompts = [target_model.tokenizer.eos_token] * args.batchsize
            inputs = target_model.tokenizer(prompts, return_tensors="pt", padding=True).to(target_model.device)

            generated_sequences = target_model.model.generate(
                input_ids = inputs.input_ids,
                attention_mask = inputs.attention_mask,
                max_length = seq_len,
                do_sample = True, 
                logits_processor = logits_warper,
                renormalize_logits = True
            )

            generated_texts = target_model.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            out_df = pd.concat([out_df, pd.DataFrame({'text':generated_texts})])

    elif args.wetfile:
        print('Seeded sampling.')
        cc_data = parse_commoncrawl(args.wetfile)

        for batch in range(num_batches):
            print(f"\nBatch number: {batch} of {num_batches}.\n")

            input_len = 10
            prompts = []
            input_ids = []
            attention_mask = []

            while len(input_ids) < args.batchsize:
                # take some random words in common crawl
                r = np.random.randint(0, len(cc_data))
                prompt = " ".join(cc_data[r:r+100].split(" ")[1:-1])

                # make sure we get the same number of tokens for each prompt to enable batching
                inputs = target_model.tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                if len(inputs['input_ids'][0]) == input_len:
                    input_ids.append(inputs['input_ids'][0])
                    attention_mask.append(inputs['attention_mask'][0])

            inputs = {'input_ids': torch.stack(input_ids).to(target_model.device), 
                        'attention_mask': torch.stack(attention_mask).to(target_model.device)}

            generated_sequences = target_model.model.generate(
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                max_length = seq_len,
                do_sample = True,
                top_k = top_k,
                top_p = 1.0
            )
            generated_texts = target_model.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            out_df = pd.concat([out_df, pd.DataFrame({'text':generated_texts})])

    else:
        print('Default sampling')
        for batch in range(num_batches):
            print(f"\nBatch number: {batch} of {num_batches}.\n")
            prompts = [target_model.tokenizer.eos_token] * args.batchsize
            inputs = target_model.tokenizer(prompts, return_tensors="pt", padding=True).to(target_model.device)
            generated_sequences = target_model.model.generate(
                input_ids = inputs.input_ids,
                attention_mask = inputs.attention_mask,
                max_length = seq_len,
                do_sample = True,
                top_k = top_k,
                top_p = 1.0
            )

            generated_texts = target_model.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            out_df = pd.concat([out_df, pd.DataFrame({'text':generated_texts})])

    out_df = out_df.drop_duplicates()
    print("\nCalculating target model perplexity.")
    out_df['target_px'] = out_df.text.apply(target_model.calculate_perplexity)
    print("Calculating lower case target model perplexity.")
    out_df['target_px_lw'] = out_df.text.str.lower().apply(target_model.calculate_perplexity)
    
    # this is an extremely expensive operation, feel free to skip.
    # print("Calculating minimum sliding window perplexity.")
    # out_df['sliding_px'] = out_df.text.apply(target_model.calculate_perplexity_sliding)

    # remove target model, free up GPU for next model.
    del target_model

    if args.reference:
         print('Evaluating reference model.')
         ref_model = LLM(args.reference)
         out_df['ref_px'] = out_df.text.apply(ref_model.calculate_perplexity)
         out_df['target_to_ref'] = out_df.target_px / out_df.ref_px

    print("Calculating zlib entropy.")
    out_df['zlib_ent'] = out_df.text.apply(do_zlib)
    out_df['target_to_zlib'] = out_df.target_px / out_df.zlib_ent
    
    out_df['target_to_lower'] = out_df.target_px / out_df.sliding_px

    out_df.to_csv(args.outfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default="EleutherAI/gpt-neo-2.7B", type=str, help='Model name to generate samples from.')
    parser.add_argument('--reference', default="EleutherAI/gpt-neo-125m", type=str, help='Model to comapre perplexity against.')
    parser.add_argument('--n', default=20, type=int, help='Number of samples to generate.')
    parser.add_argument('--batchsize', default=5, type=int, help='Batch size for generation.')
    parser.add_argument('--outfile', type=str, help='Output csv cotaining all calcualted metrics and generated text samples.')
    parser.add_argument('--wetfile', type=str, help='Path to Commoncrawl WET file')
    parser.add_argument('--dotemp', type=str, help='Do temperature decay in sample generation')

    args = parser.parse_args()

    main(args)