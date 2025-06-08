import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import os

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(img_path, max_size=400):
    try:
        image = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        raise RuntimeError(f"❌ Image file not found: {img_path}")

    size = max_size if max(image.size) > max_size else max(image.size)
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Convert tensor to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return image

# Define content and style loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Load VGG model
def get_model_and_losses(cnn, content_img, style_img):
    cnn = cnn.features.to(device).eval()
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []
    model = nn.Sequential()

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim layers after last style/content loss
    trimmed_model = nn.Sequential()
    last_loss_idx = -1
    for idx, layer in enumerate(model):
        trimmed_model.add_module(str(idx), layer)
        if isinstance(layer, ContentLoss) or isinstance(layer, StyleLoss):
            last_loss_idx = idx

    if last_loss_idx == -1:
        raise RuntimeError("❌ No content or style loss layers were added. Check your image input and layer selection.")

    trimmed_model = trimmed_model[:last_loss_idx + 1]
    return trimmed_model, style_losses, content_losses

# Main execution
def run_style_transfer():
    content = load_image("images/content.jpg")
    style = load_image("images/my_image.jpg")  # ← Updated here
    target = content.clone().requires_grad_(True)

    model, style_losses, content_losses = get_model_and_losses(models.vgg19(pretrained=True), content, style)
    optimizer = optim.LBFGS([target])

    print("🔄 Optimizing... (this might take a few minutes)")
    run = [0]
    while run[0] <= 300:
        def closure():
            optimizer.zero_grad()
            model(target)
            style_score = sum([sl.loss for sl in style_losses])
            content_score = sum([cl.loss for cl in content_losses])
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}:", f"Style Loss: {style_score.item():.4f}", f"Content Loss: {content_score.item():.4f}")

            return loss

        optimizer.step(closure)

    target.data.clamp_(0, 1)
    output = im_convert(target)
    output_image = Image.fromarray((output * 255).astype('uint8'))
    os.makedirs("output", exist_ok=True)
    output_image.save("output/stylized_image.jpg")
    print("✅ Style transfer complete! Saved to output/stylized_image.jpg")

if __name__ == '__main__':
    run_style_transfer()
