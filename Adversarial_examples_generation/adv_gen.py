import torch


# Reference: https://savan77.github.io/blog/imagenet_adv_examples.html


# Iterative FGSM method------------------------------------------------------------------
def adv_gen(noisy_img, clean_img, k, alpha, model, epsilon, loss_fn):
    patch = (clean_img != 0)
    inp = noisy_img.detach()
    inp = inp + torch.zeros_like(inp).uniform_(-epsilon, epsilon)
    for _ in range(k):
        inp.requires_grad_()
        
        with torch.enable_grad():
            out = model(inp)
            out_fin = out[patch]
            org_inputs_fin = clean_img[patch]
            loss = loss_fn(out_fin, org_inputs_fin)  
        
        grad = torch.autograd.grad(loss, [inp])[0]
        inp = inp.detach() + alpha*torch.sign(grad.detach())
        inp = torch.min(torch.max(inp, noisy_img-epsilon), noisy_img+epsilon)
        inp = torch.clamp(inp, 0, 1)
    
    return inp


