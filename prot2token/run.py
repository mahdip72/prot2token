import torch
import pandas as pd
from models import prepare_models


def test():
    torch.cuda.empty_cache()

    net, decoder_tokenizer, inference_configs  = prepare_models(name='fluorescence', device='cuda', compile_model=True)

    net.to('cuda')

    samples = pd.read_csv('../inference_data/fluorescence_inference.csv')['input'].tolist()

    results = net.run(samples[:20], merging_character='')
    print(results)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test()
    print('done!')
