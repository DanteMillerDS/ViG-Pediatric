import numpy as np
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
from model.vig_model import vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu
from model.pvig_model import pvig_ti_224_gelu, pvig_s_224_gelu, pvig_b_224_gelu, pvig_m_224_gelu

class TrainModelClassifier:
    def __init__(self, medical_type, model_name, epochs=50):
        """
        Initializes the Model with a specific medical type and computational device.
        """
        self.model_dictionary = {
            "vig_ti_224_gelu": vig_ti_224_gelu,
            "vig_s_224_gelu": vig_s_224_gelu,
            "vig_b_224_gelu": vig_b_224_gelu,
            "pvig_ti_224_gelu": pvig_ti_224_gelu,
            "pvig_s_224_gelu": pvig_s_224_gelu,
            "pvig_m_224_gelu": pvig_m_224_gelu,
            "pvig_b_224_gelu": pvig_b_224_gelu,
        }
        self.medical_type = medical_type
        self.model_name = model_name
        self.configure()
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.wd = 0.1 if self.medical_type == "ucsd" else 1e-5
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5,weight_decay = self.wd)
        self.epochs = epochs
        self.loss = nn.BCEWithLogitsLoss()
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
        self.early_stopping_patience = 10
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

    def load_model(self):
        """
        Loads a model
        :return: The model model and its associated preprocessing function.
        """
        if self.model_name is None:
            raise ValueError("Model name not specified.")
        elif self.model_name in self.model_dictionary.keys():
            model = self.model_dictionary[self.model_name]()
            model.compile()
            model.to(self.device)
        else:
            raise ValueError("Invalid model name.")
        return model

    def evaluate(self, generators, steps):
        """
        Evaluates the model using provided data loaders and computes classification metrics.
        :param generators: A dictionary of data loaders for each dataset (e.g., 'Train', 'Validation', 'Test').
        :param steps: A dictionary specifying the number of batches to evaluate for each dataset.
        :param categories: A list of categories for classification.
        :return: Accuracy, precision, recall, AUC, classification report, and confusion matrix.
        """
        y_true, y_pred, y_score = [], [], []
        self.model.eval()
        with torch.no_grad():
            for idx,(data_type, step) in enumerate(steps.items()):
                for _ in tqdm(range(step), desc=f'Evaluate {data_type}'):
                    inputs, labels = next(generators[idx])
                    inputs = torch.from_numpy(inputs.transpose(0, 3, 1,2)).to(self.device)
                    labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                    outputs = self.model(inputs)
                    predicted_probs = torch.sigmoid(outputs)
                    predicted_labels = (predicted_probs > 0.5).float()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted_labels.cpu().numpy())
                    y_score.extend(predicted_probs.cpu().numpy())
                generators[idx].reset()
        acc, prec, rec, auc = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), roc_auc_score(y_true, y_score)
        cr, cm = classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred)
        return acc, prec, rec, auc, cr, cm

    def save_results(self, acc, prec, rec, auc, cr, cm, additional_evaluation = False, additional_medical_type = False):
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
        if additional_evaluation:
            directory = f"results/zero_shot/{additional_medical_type}/{self.model_name}"
            filename = "classification_results.txt"
            filepath = os.path.join(directory, filename)
            os.makedirs(directory, exist_ok=True)
        else:
            directory = f"results/finetune/{self.medical_type}/{self.model_name}"
            filename = "classification_results.txt"
            filepath = os.path.join(directory, filename)
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "w") as file:
            file.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nAUC: {auc:.4f}\n")
            file.write(f'Classification Report\n\n{cr}\n\nConfusion Matrix\n\n{np.array2string(cm)}')
        print(f"Results saved to {filepath}")

    def train_validate(self, train_loader, validation_loader, steps):
        """
        Coordinates the training and validation of the CLIP model for a specified number of epochs.
        param train_loader: The data loader for the training dataset.
        param validation_loader: The data loader for the validation dataset.
        param steps: A dictionary specifying the number of batches to train and validate for each dataset.
        param categories: A list of categories for classification.
        """
        model_save_path = f'results/finetune/{self.medical_type}/{self.model_name}/best_model.pth'
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for _ in tqdm(range(steps["Train"]), desc=f'Epoch {epoch+1}/{self.epochs}, Train'):
                inputs, labels = next(train_loader)
                inputs = torch.from_numpy(inputs.transpose(0, 3, 1,2)).to(self.device)
                labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs,labels)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)
            self.model.eval()
            validation_losses = []
            for _ in tqdm(range(steps["Validation"]), desc=f'Epoch {epoch+1}/{self.epochs}, Validation'):
                inputs, labels = next(validation_loader)
                inputs = torch.from_numpy(inputs.transpose(0, 3, 1,2)).to(self.device)
                labels = torch.from_numpy(labels).to(self.device).float().unsqueeze(1)
                outputs = self.model(inputs)
                loss = self.loss(outputs,labels)
                validation_losses.append(loss.item())
            avg_validation_loss = np.mean(validation_losses)
            train_acc, train_prec, train_rec, train_auc, _, _ = self.evaluate([train_loader], {"Train":steps["Train"]})
            val_acc, val_prec, val_rec, val_auc, _, _ = self.evaluate([validation_loader], {"Validation":steps["Validation"]})
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
                plt.savefig(f'results/finetune/{self.medical_type}/{self.model_name}/metrics_{metric_name}_epoch.png')
                plt.close()

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, AUC: {train_auc:.4f}")
            print(f"Val - Loss: {avg_validation_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, AUC: {val_auc:.4f}")
            if avg_validation_loss < self.best_val_loss:
                self.best_val_loss = avg_validation_loss
                torch.save(self.model.state_dict(), model_save_path)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter == self.early_stopping_patience:
                    self.early_stop = True
                    print("Early stopping triggered.")
                    break
        self.model.load_state_dict(torch.load(model_save_path))

    def run(self, generators, steps):
        """
        Performs finetuning of a model on the train and validation data laoder then evalautes on the testing dataframe.
        :param generators: A dictionary of data loaders for each dataset.
        :param steps: A dictionary specifying the number of batches to evaluate for each dataset.
        :param categories: A list of categories for classification.
        :return: None. Prints the evaluation metrics and saves the results.
        """
        self.train_validate(generators[0],generators[1],steps)
        acc, prec, rec, auc, cr, cm = self.evaluate([generators[2]], {"Test": steps["Test"]})
        print(f"\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
        self.save_results(acc, prec, rec, auc, cr, cm)