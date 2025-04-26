# Rock Paper Scissors Sentiment Analysis ✊🖐️✌️🧠  
A convolutional neural network (CNN) project built using PyTorch to classify hand gesture images (rock, paper, scissors) using the Rock-Paper-Scissors image dataset. The model aims to learn gesture patterns and make accurate classifications based on visual input.

## Introduction 🌍  
Hand gesture recognition is a foundational task in computer vision with applications in games, robotics, sign language recognition, and human-computer interaction. This project demonstrates how to train a CNN using PyTorch to classify images of hand gestures into three categories: **rock**, **paper**, or **scissors**.

By building the full pipeline from data loading to model evaluation, this project provides a simple but solid foundation for gesture-based image classification using deep learning.

## Dataset Overview 📊  
The Rock-Paper-Scissors dataset contains labeled images of hand gestures (rock, paper, or scissors) in various positions and lighting conditions, ideal for training vision models.

### Classes
- **Rock**
- **Paper**
- **Scissors**

Dataset Source: [Rock-Paper-Scissors on Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors?utm_source=chatgpt.com&select=scissors)

## Methodology 🔍  
1. **Data Loading & Transformation**: 
   - Images are converted to tensors and normalized for consistent input.
2. **Data Preparation**: 
   - Use `DataLoader` to batch and shuffle training data.
3. **CNN Architecture**: 
   - Two convolutional layers followed by three fully connected layers.
4. **Loss & Optimization**: 
   - CrossEntropyLoss with SGD optimizer and momentum.
5. **Model Training**:
   - Train the model for 2 epochs using a small batch size.
6. **Saving Model**:
   - Save trained weights to a `.pth` file.
7. **Evaluation**:
   - Load saved model weights and measure accuracy on test data.

## Project Objectives 🎯  
Develop a functional image classification model that can:
- Distinguish between rock, paper, and scissors gestures.
- Serve as a baseline for more complex gesture recognition systems.
- Provide hands-on experience with CNNs using PyTorch.

## Model Performance 📈  
- **Epochs Trained**: 2  
- **Accuracy on Test Set**: ~54%  
- **Loss Function**: Cross Entropy  
- **Optimizer**: SGD (momentum 0.9)  
- **Observation**: Basic architecture achieves moderate accuracy, indicating potential for deeper architectures and more training epochs.

## Strengths 💪  
1. **Simple and educational**: Good introduction to CNNs and image classification.  
2. **Fast to train**: Lightweight setup suitable for CPU training.  
3. **Easy to extend**: Code is modular and beginner-friendly.

## Weaknesses ⚠️  
1. **Low epoch count**: Only 2 epochs limits learning.  
2. **No regularization or augmentation**: Susceptible to underfitting or overfitting.  
3. **Simple architecture**: Deeper models likely perform better.

## Potential Improvements 🔧  
- Train for more epochs and experiment with batch sizes.  
- Use data augmentation (flip, rotate, brightness changes).  
- Add dropout or batch normalization layers.  
- Try advanced optimizers (Adam, RMSprop).  
- Use a pre-trained model or transfer learning.

## Libraries Used 🛠️  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib

## Model Deployment 🚀  
Try the model live on Hugging Face Spaces: [Rock Paper Scissors Sentiment Analysis](https://huggingface.co/spaces/Skyrezz/rock_paper_scissors_sentiment_analysis)

## Author 👨‍💻  
Reza Syadewo  
LinkedIn: [https://www.linkedin.com/in/reza-syadewo-b5801421b/](https://www.linkedin.com/in/reza-syadewo-b5801421b/)