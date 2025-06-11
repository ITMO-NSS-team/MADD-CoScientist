import pickle as pi
from scripts.model import MolGen


# steps = 8 or 10 good for ds size 1700000 samples
def run(path_ds: str, lr: float = 0.0003, bs: int = 256, steps: int = 10, hidden: int = 256):
    data = []
    with open(path_ds, "r") as f:
        for line in f.readlines()[1:]:
            data.append(line.split("\n")[0])


    # create model
    gan_mol = MolGen(data, hidden_dim=hidden, lr=lr, device="cuda")
    # create dataloader
    loader = gan_mol.create_dataloader(data, batch_size=bs, shuffle=True, num_workers=0)
    # train model for 10000 steps
    gan_mol.train_n_steps(loader, max_step=steps, evaluate_every=150)
    # save model
    pi.dump(gan_mol, open('v1_gan_mol_{0}_{1}_{2}k.pkl'.format(bs, lr, steps // 1000), 'wb'))

# run train LSTM GAN
if __name__ == '__main__':
    path = 'generative_models/GAN/gan_lstm_refactoring/chembl_filtered_400w_150s.csv' #change if another
    run(path)