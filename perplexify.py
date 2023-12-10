import argparse
import pandas as pd
from llm import LLM
import torch

def main(args):
    print("Loading model...")
    ref_model = LLM(args.reference)
    model_name = '_'.join(str(args.reference).split('/'))
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    out_df = pd.read_csv(args.infile)

    print('Evaluating reference model.')
    out_df[model_name] = [ref_model.calculate_perplexity(str(x)) for x in out_df.text.to_list()]
    out_df['target_to_{ref}'.format(ref=model_name)] = out_df.target_px / out_df[model_name]

    out_df.to_csv(args.outfile, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', default="EleutherAI/gpt-neo-2.7B", type=str, help='Model to comapre perplexity against.')
    parser.add_argument('-i', '--infile', type=str, help='Input csv from infer.py')
    parser.add_argument('-o', '--outfile', type=str, help='Output csv cotaining all calcualted metrics and generated text samples.')
    args = parser.parse_args()

    main(args)
