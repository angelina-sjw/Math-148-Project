import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from data_utils.utils import preprocess_image, unnormalize_image


class GradCAM():
    """
    Implements the Grad-CAM visualization technique for CNN-based models.

    Args:
        model (nn.Module): The neural network model.
        target_layer (nn.Module): The layer from which gradients and activations are extracted.

    Attributes:
        gradients (Tensor): Stores gradients of the target layer during backpropagation.
        activations (Tensor): Stores activations of the target layer during forward pass.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_layer()

    def hook_layer(self):
        """
        Hooks forward and backward passes to capture activations and gradients.
        """
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generates a Grad-CAM heatmap for a given input image.

        Args:
            input_tensor (Tensor): Preprocessed input image tensor.
            target_class (int, optional): Target class index for computing CAM. 
                                          Defaults to the predicted class.

        Returns:
            np.ndarray: Normalized Grad-CAM heatmap.
        """
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax().item()
        
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    @staticmethod
    def overlay_heatmap(image, cam, alpha=0.5):
        """
        Overlays a Grad-CAM heatmap on the original image.

        Args:
            image (np.ndarray): Original image in RGB format.
            cam (np.ndarray): Grad-CAM heatmap.
            alpha (float): Transparency level for the heatmap overlay.

        Returns:
            np.ndarray: Image with heatmap overlay.
        """
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay


def display_grad_cam(model, device, photo_ids, photo_dir, num_images=5):
    """
    Displays Grad-CAM visualizations for a set of images.

    Args:
        model (nn.Module): The trained model.
        device (torch.device): Device to run inference on (CPU/GPU).
        photo_ids (list): List of image IDs.
        photo_dir (str): Directory where images are stored.
        num_images (int): Number of images to display.
    """
    selected_photo_ids = random.sample(list(photo_ids), num_images)
    fig, axes = plt.subplots(2, num_images, figsize=(15, 10), constrained_layout=True)
    for ax, photo_id in zip(axes[0], selected_photo_ids):
        image_path = f"{photo_dir}/{photo_id}.jpg"
        original_image, _ = preprocess_image(image_path)
        original_image_np = np.array(original_image)
        ax.imshow(original_image_np)
        ax.set_title(f"{photo_id}")
        ax.axis("off")

    for ax, photo_id in zip(axes[1], selected_photo_ids):
        image_path = f"{photo_dir}/{photo_id}.jpg"
        original_image, input_tensor = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)

        target_layer = model.base.layer4[1] 
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor)
        overlay_image = gradcam.overlay_heatmap(np.array(original_image), cam)

        ax.imshow(overlay_image)
        ax.set_title(f"{photo_id}")
        ax.axis("off")

    plt.show()


def display_target_image_grad_cam(model, device, image_tensor):
    """
    Applies Grad-CAM to the given input tensor using the specified model.

    Args:
        model (nn.Module): The trained model.
        device (torch.device): Device to run inference on (CPU/GPU).
        input_tensor (torch.Tensor): Preprocessed input image tensor (batch size can be 1 or more).

    Returns:
        np.ndarray: The Grad-CAM heatmap (2D) for the input tensor.
    """
    # Make sure model is in eval mode
    model.eval()
    
    # Send the tensor to the correct device
    input_tensor = input_tensor.to(device)

    # Select the layer you want Grad-CAM to target
    target_layer = model.base.layer4[1]
    
    # Initialize your GradCAM object
    gradcam = GradCAM(model, target_layer)
    
    # Generate the Grad-CAM heatmap
    cam = gradcam.generate_cam(input_tensor)
    
    return cam


def display_target_image_grad_cam(model, device, input_tensor):
    """
    Applies Grad-CAM to the given input tensor using the specified model.

    Args:
        model (nn.Module): The trained model.
        device (torch.device): Device to run inference on (CPU/GPU).
        input_tensor (torch.Tensor): Preprocessed input image tensor (batch size can be 1 or more).

    Returns:
        np.ndarray: The Grad-CAM heatmap (2D) for the input tensor.
    """
    # Make sure model is in eval mode
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)

    target_layer = model.base.layer4[1]

    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate_cam(input_tensor)

    unnorm_image = unnormalize_image(input_tensor)

    overlay_image = gradcam.overlay_heatmap(unnorm_image, cam)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_image)
    plt.axis("off")
    plt.show()