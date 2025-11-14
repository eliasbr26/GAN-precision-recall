import torch
import os



def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G_60000.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, device):
    ckpt_path = os.path.join(folder,'G_60000.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G


def estimate_D_M(G, D, num_samples=10000, batch_size=128):
    G.eval(); D.eval()
    max_logit = -float('inf')
    
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            z = torch.randn(batch_size, 100).cuda()
            x_fake = G(z)
            D_logit = D(x_fake).squeeze()  # Logits (avant sigmoïde)
            max_logit = max(max_logit, D_logit.max().item())
    
    G.train(); D.train()
    return max_logit



def generate_samples_with_full_DRS(G, D, num_samples, batch_size=128, epsilon=1e-6, target_percentile=70):
    G.eval(); D.eval()
    samples = []
    total_generated = 0
    total_attempted = 0

    # 1. Estimer D_M
    D_M = estimate_D_M(G, D, num_samples=10000, batch_size=batch_size)
    print(f"Estimated D_M: {D_M:.4f}")

    while total_generated < num_samples:
        z = torch.randn(batch_size, 100).cuda()
        with torch.no_grad():
            x_fake = G(z)
            D_logits = D(x_fake).squeeze()  # logit

            # 2. Calculer F(x)
            delta = D_logits - D_M
            # Clamp pour éviter que exp(delta) > 1
   
            F_x = D_logits - torch.log(1 - torch.exp(delta) + epsilon)


            # 3. Estimer gamma dynamiquement (80e percentile du batch)
            gamma = torch.quantile(F_x, 1- target_percentile / 100.0).item()

            # 4. Calculer la probabilité d’acceptation
            F_hat = F_x - gamma
            acceptance_probs = torch.sigmoid(F_hat)

            # 5. Tirage d’acceptation
            accept = torch.bernoulli(acceptance_probs).bool()
            accepted_samples = x_fake[accept]
            samples.append(accepted_samples.cpu())

            total_generated += accepted_samples.size(0)
            total_attempted += batch_size

    acceptance_rate = total_generated / total_attempted
    print(f"Acceptance Rate: {acceptance_rate:.4f}")

    G.train(); D.train()

    samples = torch.cat(samples, dim=0)[:num_samples]
    return samples



def generate_samples_with_DRS(G, D, num_samples, batch_size, tau):
    G.eval()
    D.eval()
    samples = []
    total_generated = 0
    total_attempted = 0

    while total_generated < num_samples:
        # Generate latent vectors
        z = torch.randn(batch_size, 100).cuda()
        with torch.no_grad():
            # Generate samples
            x_fake = G(z)
            # Compute discriminator logits
            D_output = D(x_fake).squeeze()
            # Compute acceptance probabilities
            acceptance_probs = torch.sigmoid(D_output - tau)
            # Sample from Bernoulli distribution
            accept = torch.bernoulli(acceptance_probs).bool()
            # Select accepted samples
            accepted_samples = x_fake[accept]
            samples.append(accepted_samples.cpu())
            total_generated += accepted_samples.size(0)
            total_attempted += batch_size
        
            # if total_generated != 0  and total_generated % 100 == 0:
            #     print(f'Generated {total_generated}/{num_samples} samples')

    acceptance_rate = total_generated / total_attempted
    print(f'Acceptance Rate: {acceptance_rate:.4f}')

    G.train()
    D.train()

    # Concatenate all accepted samples
    samples = torch.cat(samples, dim=0)
    # If more samples than needed, truncate
    samples = samples[:num_samples]

    return samples












