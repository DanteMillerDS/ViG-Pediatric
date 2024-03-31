import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import data_loader.extract_data as extract_data
import data_loader.load_data as load_data
import visualize.visualize as visualize
from fine_tune.finetune import TrainModelClassifier
import numpy as np
def run_finetune_vig_b_224_gelu(medical_type, model_type, batch_size, additional_evaluation = False, mean_and_std = False):
    """
    Performs fine-tuning of a vig_b_224_gelu model on a given medical type and model type.
    :param medical_type: The type of medical data to classify ('ucsd', 'ori').
    :param model_type: The type of model to use for classification ().
    :param batch_size: The batch size for data loading.
    """
    generators, lengths = load_data.create_loader(medical_type, batch_size,finetune = True, mean_and_std = mean_and_std)
    visualize.save_random_images_from_generators(generators, [medical_type, model_type, "finetune"], 2)
    if model_type == "vig_b_224_gelu":
        classifier = TrainModelClassifier(medical_type,model_type, mean_and_std = mean_and_std)
        classifier.run(generators, lengths)
        if additional_evaluation:
            generators, lengths = load_data.create_loader(additional_evaluation, batch_size,finetune = False, mean_and_std = mean_and_std)
            steps = {"Train":lengths["Train"], "Validation":lengths["Validation"], "Test":lengths["Test"]}
            acc, prec, rec, auc, cr, cm = classifier.evaluate(generators, steps)
            classifier.save_results(acc, prec, rec, auc, cr, cm, additional_evaluation = True, additional_medical_type = additional_evaluation)
            
        return classifier
    else:
        print("Did not define a proper classifer!")

if __name__ == "__main__":
    extract_data.mount_and_process()
    mean_and_std = {"ori": {"mean": np.array([0.4856, 0.4856, 0.4856]), "std": np.array([0.2463, 0.2463, 0.2463])}, "ucsd": {"mean": np.array([0.4655, 0.4655, 0.4655]), "std": np.array([0.2523, 0.2523, 0.2523])}}
    batch_size = 128 
    model_types = ['vig_b_224_gelu']
    medical_types = ['ucsd', 'ori']
    ucsd_classifier = run_finetune_vig_b_224_gelu(medical_types[0], model_types[0], batch_size, additional_evaluation=medical_types[1], mean_and_std = mean_and_std[medical_types[0]])
    ori_classifier = run_finetune_vig_b_224_gelu(medical_types[1], model_types[0], batch_size, additional_evaluation=medical_types[0],mean_and_std = mean_and_std[medical_types[1]])