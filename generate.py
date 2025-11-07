import torch 
import torchvision
import os
import argparse


from model import Generator, Discriminator  
from utils import load_model, generate_samples_with_full_DRS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()



    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    G = Generator(g_output_dim=mnist_dim).to(device)
    G.load_state_dict(torch.load('checkpoints/G_60000.pth', map_location=device))
    D = Discriminator(d_input_dim=mnist_dim).to(device)
    D.load_state_dict(torch.load('checkpoints/D_60000.pth', map_location=device))

    if torch.cuda.device_count() > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)

    samples = generate_samples_with_full_DRS(G, D, num_samples=10000, batch_size=args.batch_size)

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    for idx in range(samples.size(0)):
        torchvision.utils.save_image(samples[idx].view(1, 28, 28),
                          f'samples/{idx}.png',
                          normalize=True)


    
