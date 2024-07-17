
# CIFAR-10 Image Classification Telegram Bot

This Telegram bot classifies images using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The bot can train the model and then use it to predict the class of objects in user-uploaded photos.

## Features

- `/start` - Starts the conversation and provides a list of available commands.
- `/help` - Shows the help message with available commands.
- `/train` - Trains the neural network model using the CIFAR-10 dataset.
- Handles text messages by prompting the user to train the model and send a picture.
- Handles photo messages by predicting the class of the uploaded image.

## Prerequisites

- Python 3.6 or higher
- Telegram Bot API token

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/cifar10-telegram-bot.git
    cd cifar10-telegram-bot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a file named `token.txt` in the root directory of the project and paste your Telegram bot API token into it:
  

## Usage

1. Run the bot:
    ```sh
    python bot.py
    ```

2. Open Telegram and start a conversation with your bot. Use the command `/start` to see the available commands.

3. Train the model by sending the `/train` command.

4. After the training is complete, send a photo to the bot, and it will reply with the predicted class of the object in the photo.

## Dependencies

- `python-telegram-bot`
- `tensorflow`
- `opencv-python`
- `numpy`

## Logging

The bot uses the `logging` module to log information and errors. Logs are printed to the console.

## Model

The bot uses a simple CNN model with the following architecture:
- 3 Convolutional layers with ReLU activation and MaxPooling
- Flatten layer
- 2 Dense layers with ReLU activation
- Output Dense layer with 10 units (for 10 classes) and no activation (logits)

## Note

- The bot currently supports only the 'Car' class from the CIFAR-10 dataset for simplicity. You can modify the `class_names` variable to include other classes as needed.
- The model is saved to a file named `cifar_classifier.keras` after training.

## License

This project is licensed under the MIT License.
