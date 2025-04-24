# dump_inits.py
import torch 
import random
import numpy as np
from holographic_vae.cg_coefficients.get_w3j_coefficients import get_w3j_coefficients

def load_w3j(lmax=7):
    w3j_np = get_w3j_coefficients(lmax)
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in w3j_np.items()}

# your five different model classes:
from ML_Model.model    import So3HyperCNNv2 as Model1
from ML_Model_2.model  import So3HyperCNNv2 as Model2
from ML_Model_3.model  import ConvHyperCNN  as Model3
from ML_Model_4.model  import ConvHyperCNN    as Model4

# reproducibility
SEED = 46
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
w3j = load_w3j(7)

# instantiate each with the same seed/hyperparams
m1 = Model1(w3j, num_freq_bins=512, latent_dim=128, anthro_dim=25).to(device)
m2 = Model2(w3j, num_freq_bins=512, latent_dim=128, anthro_dim=25).to(device)
m3 = Model3(num_freq_bins=512, latent_dim=128, anthro_input_dim=25).to(device)
m4 = Model4(num_freq_bins=512, latent_dim=128, anthro_input_dim=25).to(device)

# save them all
for idx, model in enumerate((m1, m2, m3, m4), start=1):
    torch.save(model.state_dict(), f"model{idx}_init.pth")

print("✅ dumped 4 init checkpoints")


