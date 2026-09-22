# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

This code implements a Denoising Autoencoder using PyTorch to clean noisy images from the MNIST dataset. It uses a convolutional neural network architecture, where the encoder compresses the input image into a lower-dimensional representation, and the decoder reconstructs the original image from this compressed form. To train the model to remove noise, Gaussian noise is added to the clean images, and the network learns to recover the original from the noisy version. The training process uses Mean Squared Error (MSE) as the loss function to measure the reconstruction error and the Adam optimizer to update the model weights. The autoencoder is trained over multiple epochs using mini-batches of data for efficiency. After training, the model's performance is visually evaluated by displaying the original, noisy, and denoised images side by side.

<img width="642" height="483" alt="image" src="https://github.com/user-attachments/assets/e74891a3-eac2-45bd-9bd4-34d04e91e9ff" />


## DESIGN STEPS
### STEP 1: 

Load MNIST data and add noise to images.



### STEP 2: 

Build a convolutional autoencoder.



### STEP 3: 

Train the model with noisy images, minimizing MSE loss.



### STEP 4: 

Update weights using backpropagation.



### STEP 5: 

Test the model and visualize original, noisy, and denoised images.



### STEP 6: 


Repeat through multiple epochs for better denoising performance.




## PROGRAM

### Name: Chandru M

### Register Number: 212224230041

```python
# Denoising Autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# Initialize model
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training function
 def train(model, loader, criterion, optimizer, epochs=5):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")


# Visualization function


def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Chandru M ")
    print("Register Number: 212224230041 ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


```

### OUTPUT

### Model Summary


<img width="1267" height="469" alt="image" src="https://github.com/user-attachments/assets/3b6420dc-12fc-473c-af71-e8e6be7ae997" />


### Training loss

<img width="461" height="117" alt="image" src="https://github.com/user-attachments/assets/53f05c19-1398-4f2b-af91-a287a5a79d2e" />


## Original vs Noisy Vs Reconstructed Image


<img width="1789" height="589" alt="image" src="https://github.com/user-attachments/assets/3d6b2082-607a-4bf6-974b-2544caa60e45" />


## RESULT

Thus, a convolutional autoencoder for image denoising application has been developed.


