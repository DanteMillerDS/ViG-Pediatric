import numpy as np
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import clip
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt

class TrainClipClassifier:
    def __init__(self, medical_type, epochs=50):
        """
        Initializes the CLIPZeroShotClassifier with a specific medical type and computational device.
        """
        self.medical_type = medical_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.configure()
        self.clip_model, self.preprocess = self.load_clip_model()
        self.wd = 0.1 if self.medical_type == "ucsd" else 1e-5
        self.optimizer = optim.Adam(self.clip_model.parameters(), lr=1e-5,weight_decay = self.wd)
        self.epochs = epochs
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.metric_history  = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_auc': [],
            'val_auc': [],
        }
        self.early_stopping_patience = 20
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False

    def configure(self):
        """
        Configures the system environment for optimal performance.
        :return: None. Prints the status of NVIDIA library configuration.
        """
        status = os.system('ldconfig /usr/lib64-nvidia')
        if status == 0:
            print("NVIDIA library configured successfully.")
        else:
            print("Error configuring NVIDIA library.")
        torch._dynamo.config.suppress_errors = True

    def convert_models_to_fp32(self, model):
        """
        Converts model parameters and gradients to float32 precision. This is necessary for compatibility with certain optimizers or hardware.
        :params model: The model to convert to float32 precision.
        :return: None. Converts the model parameters and gradients in-place.
        """   
        for p in model.parameters():
            if p.grad is not None:
                p.data = p.data.float()
                p.grad.data = p.grad.data.float()

    def load_clip_model(self):
        """
        Loads the CLIP model and preprocessing function into the specified device.
        :return: The CLIP model and its associated preprocessing function.
        """
        model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        if self.device == "cpu":
            model.float()
        else :
            clip.model.convert_weights(model)
        return model, preprocess

    def zero_shot_classification(self, image_batch, categories):
        """
        Performs zero-shot classification using the CLIP model on a batch of images.
        :param image_batch: A tensor representing a batch of images.
        :param categories: A list of categories for classification.
        :return: The top probabilities and labels for the classification predictions.
        """
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c} lungs.") for c in categories]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_batch)
            text_features = self.clip_model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = similarity.topk(1, dim=-1)
        return top_probs, top_labels

    def evaluate(self, generators, steps, categories):
        """
        Evaluates the CLIP model using provided data loaders and computes classification metrics.
        :param generators: A dictionary of data loaders for each dataset (e.g., 'Train', 'Validation', 'Test').
        :param steps: A dictionary specifying the number of batches to evaluate for each dataset.
        :param categories: A list of categories for classification.
        :return: Accuracy, precision, recall, AUC, classification report, and confusion matrix.
        """
        y_true, y_pred, y_score = [], [], []
        self.clip_model.eval()
        with torch.no_grad():
            for idx,(data_type, step) in enumerate(steps.items()):
                for _ in tqdm(range(step), desc=f'Evaluate {data_type}'):
                    inputs, labels = next(generators[idx])
                    inputs = torch.from_numpy(inputs.transpose(0, 3, 1,2)).to(self.device)
                    labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                    top_probs, top_labels = self.zero_shot_classification(inputs, categories)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(top_labels.cpu().numpy())
                    y_score.extend(top_probs.cpu().numpy())
                generators[idx].reset()
        acc, prec, rec, auc = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), roc_auc_score(y_true, y_score)
        cr, cm = classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred)
        return acc, prec, rec, auc, cr, cm

    def save_results(self, acc, prec, rec, auc, cr, cm):
        """
        Saves the evaluation results to a file within a directory specific to the medical type and CLIP model.
        :param acc: The accuracy of the classification.
        :param prec: The precision of the classification.
        :param rec: The recall of the classification.
        :param auc: The AUC of the classification.
        :param cr: The classification report.
        :param cm: The confusion matrix.
        :return: None. Results are saved to a text file.
        """
        directory = f"results/t_pretrained/{self.medical_type}/clip"
        filename = "classification_results.txt"
        filepath = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)
        with open(filepath, "w") as file:
            file.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nAUC: {auc:.4f}\n")
            file.write(f'Classification Report\n\n{cr}\n\nConfusion Matrix\n\n{np.array2string(cm)}')
        print(f"Results saved to {filepath}")

    def train_validate(self, train_loader, validation_loader, steps, categories):
        """
        Coordinates the training and validation of the CLIP model for a specified number of epochs.
        param train_loader: The data loader for the training dataset.
        param validation_loader: The data loader for the validation dataset.
        param steps: A dictionary specifying the number of batches to train and validate for each dataset.
        param categories: A list of categories for classification.
        """
        model_save_path = f'results/t_pretrained/{self.medical_type}/clip/best_model.pth'
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        for epoch in range(self.epochs):
            self.clip_model.train()
            train_losses = []
            for _ in tqdm(range(steps["Train"]), desc=f'Epoch {epoch+1}/{self.epochs}, Train'):
                inputs, labels = next(train_loader)
                inputs = torch.from_numpy(inputs.transpose(0, 3, 1,2)).to(self.device)
                labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                texts = torch.cat([clip.tokenize(f"a photo of {categories[int(label.item())]} lungs.") for label in labels]).to(self.device)
                self.optimizer.zero_grad()
                logits_per_image, logits_per_text = self.clip_model(inputs, texts)
                ground_truth = torch.arange(len(inputs),dtype=torch.long,device=self.device)
                total_loss = (self.loss_img(logits_per_image,ground_truth) + self.loss_txt(logits_per_text,ground_truth))/2
                total_loss.backward()
                if self.device == "cpu":
                    self.optimizer.step()
                else :
                    self.convert_models_to_fp32(self.clip_model)
                    self.optimizer.step()
                    clip.model.convert_weights(self.clip_model)
                train_losses.append(total_loss.item())
            avg_train_loss = np.mean(train_losses)
            self.clip_model.eval()
            validation_losses = []
            for _ in tqdm(range(steps["Validation"]), desc=f'Epoch {epoch+1}/{self.epochs}, Validation'):
                inputs, labels = next(validation_loader)
                inputs = torch.from_numpy(inputs.transpose(0, 3, 1,2)).to(self.device)
                labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                texts = torch.cat([clip.tokenize(f"a photo of {categories[int(label.item())]} lungs.") for label in labels]).to(self.device)
                logits_per_image, logits_per_text = self.clip_model(inputs, texts)
                ground_truth = torch.arange(len(inputs),dtype=torch.long,device=self.device)
                total_loss = (self.loss_img(logits_per_image,ground_truth) + self.loss_txt(logits_per_text,ground_truth))/2
                validation_losses.append(total_loss.item())
            avg_validation_loss = np.mean(validation_losses)
            train_acc, train_prec, train_rec, train_auc, _, _ = self.evaluate([train_loader], {"Train":steps["Train"]}, categories)
            val_acc, val_prec, val_rec, val_auc, _, _ = self.evaluate([validation_loader], {"Validation":steps["Validation"]}, categories)
            self.metric_history['train_loss'].append(avg_train_loss)
            self.metric_history['val_loss'].append(avg_validation_loss)
            self.metric_history['train_accuracy'].append(train_acc)
            self.metric_history['val_accuracy'].append(val_acc)
            self.metric_history['train_precision'].append(train_prec)
            self.metric_history['val_precision'].append(val_prec)
            self.metric_history['train_recall'].append(train_rec)
            self.metric_history['val_recall'].append(val_rec)
            self.metric_history['train_auc'].append(train_auc)
            self.metric_history['val_auc'].append(val_auc)

            epochs_range = range(1, epoch + 2)
            for i, (metric_name) in enumerate(['loss', 'accuracy', 'precision', 'recall', 'auc'], 1):
                plt.figure(figsize=(10, 6))
                plt.plot(epochs_range, self.metric_history[f'train_{metric_name}'], label=f'Train {metric_name.capitalize()}')
                plt.plot(epochs_range, self.metric_history[f'val_{metric_name}'], label=f'Validation {metric_name.capitalize()}', linestyle='--')
                plt.legend(loc='best')
                plt.title(metric_name.capitalize())
                plt.tight_layout()
                plt.savefig(f'results/t_pretrained/{self.medical_type}/clip/metrics_{metric_name}_epoch.png')
                plt.close()

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, AUC: {train_auc:.4f}")
            print(f"Val - Loss: {avg_validation_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, AUC: {val_auc:.4f}")
            if avg_validation_loss < self.best_val_loss:
                self.best_val_loss = avg_validation_loss
                torch.save(self.clip_model.state_dict(), model_save_path)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter == self.early_stopping_patience:
                    self.early_stop = True
                    print("Early stopping triggered.")
                    break
        self.clip_model.load_state_dict(torch.load(model_save_path))


    def run(self, generators, steps, categories = ['normal', 'covid']):
        """
        Coordinates the process of zero-shot classification evaluation and result saving for the CLIP model.
        :param generators: A dictionary of data loaders for each dataset.
        :param steps: A dictionary specifying the number of batches to evaluate for each dataset.
        :param categories: A list of categories for classification.
        :return: None. Prints the evaluation metrics and saves the results.
        """
        self.train_validate(generators[0],generators[1],steps,categories)
        acc, prec, rec, auc, cr, cm = self.evaluate([generators[2]], {"Test":steps["Test"]}, categories)
        print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
        self.save_results(acc, prec, rec, auc, cr, cm)